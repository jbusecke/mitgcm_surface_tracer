from __future__ import print_function
import xarray as xr
import numpy as np
import time
import os
import os.path
from xmitgcm import open_mdsdataset
from xarrayutils.numpy_utils import interp_map_regular_grid
from .utils import readbin, writebin


def validmask_aviso(da_u, da_v):
    validmask_aviso_u = np.all(~np.isnan(da_u),
                               axis=da_u.get_axis_num('time'))
    validmask_aviso_v = np.all(~np.isnan(da_v),
                               axis=da_v.get_axis_num('time'))

    if np.logical_xor(validmask_aviso_u, validmask_aviso_v).any():
        raise RuntimeWarning('U and V validmask are not equal')

    validmask_aviso = np.logical_and(validmask_aviso_u, validmask_aviso_v)
    return validmask_aviso


def combine_validmask(data_dir, shape=None, debug=False):

    fnames = []
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for filename in [f for f in filenames if f == 'validmask.bin']:
            print('found validmask at '+os.path.join(dirpath, filename))
            fnames.append(os.path.join(dirpath, filename))
    if debug:
        print('data_dir', data_dir)
        print(fnames)

    if shape:
        masks = np.array([readbin(f, shape) for f in fnames])
    else:
        raise RuntimeWarning('When shape is not given')

    combo = np.all(np.stack(masks, axis=2), axis=2)

    fpath = data_dir+'/validmask_combined.bin'
    writebin(combo, fpath)
    print('--- combined validmask written to '+fpath+' ---')


def process_aviso(odir, gdir, ddir_dt, fid_dt,
                  ddir_nrt=None, fid_nrt=None, interpolate=True):
    """read aviso files into xarray dataset, respecting 'seam' between
    delayed-time
    product and near-real time products

    """
    grid = open_mdsdataset(gdir, iters=None)
    ds_dt = xr.open_mfdataset(ddir_dt+'/'+fid_dt).drop(['nv', 'crs'])
    if ddir_nrt is not None:
        transition_date = ds_dt.time.isel(time=-1)
        ds_nrt = xr.open_mfdataset(ddir_nrt+'/'+fid_nrt).drop(['nv', 'crs'])
        ds = xr.concat((ds_dt,
                        ds_nrt.isel(time=ds_nrt.time > transition_date)),
                       dim='time')
        # Test if time is continous
        if np.any(ds.time.diff('time').data != ds.time.diff('time')[0].data):
            print(ds.time)
            raise RuntimeError('Time steps are not homogeneous. Likely missing \
            files between the dt and nrt products')
    else:
        ds = ds_dt
        transition_date = ''
    start_date = ds.time[0].data

    # Write out the startdate and transition_date to textfile
    f = open(odir+'/startdate.txt', 'w')
    f.write(str(start_date))
    f.close()
    print('--- startdate written to '+odir+'/startdate.txt ---')

    f = open(odir+'/transitiondate.txt', 'w')
    f.write(str(transition_date))
    f.close()
    print('--- startdate written to '+odir+'/startdate.txt ---')

    # create and save validmask
    # validmask indicates values that were interpolated or filled
    # and should be taken out for certain interpretations.
    validmask = validmask_aviso(ds.u, ds.v)
    validmask_mit = interp_map_regular_grid(validmask,
                                            ds.u.lon.data,
                                            ds.u.lat.data,
                                            grid.XC.data,
                                            grid.YC.data)
    # fill the values between 0 and with 1 to exclude them from valid points.
    validmask_mit[validmask_mit != 1] = 0
    writebin(validmask_mit, odir+'/validmask.bin')
    print('--- validmask written to '+odir+'/validmask.bin ---')

    start_time = time.time()
    for tt, ti in enumerate(ds.time.data):
        u = ds.u.sel(time=ti).values
        v = ds.v.sel(time=ti).values

        # Velocities near the coast are padded with zeros
        u[np.isnan(u)] = 0
        v[np.isnan(v)] = 0

        u_int = interp_map_regular_grid(u,
                                        ds.u.lon.data,
                                        ds.u.lat.data,
                                        grid.XG.data,
                                        grid.YC.data)
        v_int = interp_map_regular_grid(v,
                                        ds.v.lon.data,
                                        ds.v.lat.data,
                                        grid.XC.data,
                                        grid.YG.data)

        if np.any(np.isnan(u_int.flatten())):
            raise RuntimeError('Nans detected in the u fields')
        if np.any(np.isnan(v_int.flatten())):
            raise RuntimeError('Nans detected in the v fields')

        writebin(u_int, odir+'/uvel'+str(ti))
        writebin(v_int, odir+'/vvel'+str(ti))

        print(str(tt)+'/'+str(len(ds.time)))
        print(str(ti))

    print("--- Velocity Interpolation took %s seconds ---"
          % (time.time() - start_time))
