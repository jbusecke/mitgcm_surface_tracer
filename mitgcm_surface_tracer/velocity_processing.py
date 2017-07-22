from __future__ import print_function
import xarray as xr
import numpy as np
import time
import os
import os.path
from xmitgcm import open_mdsdataset
from xarrayutils.numpy_utils import interp_map_regular_grid
from .utils import readbin, writebin, writetxt
from dask.diagnostics import ProgressBar


def merge_aviso(ddir_dt,
                fid_dt='dt_global_allsat_msla_uv_*.nc',
                ddir_nrt=None,
                fid_nrt='nrt_global_allsat_msla_uv_*.nc'):

    """read aviso files into xarray dataset
    This function merges delayed-time and near-real time products if optional
    near-real time parameters are given.

    PARAMETERS
    ----------
    ddir_dt : path
        data directory for delayed time product
    fid_dt : str
        string pattern identifying delayed time products
        (default:'dt_global_allsat_msla_uv_*.nc')
    ddir_dt : path
        data directory for near-real time product
        (default: None)
    fid_dt : str
        string pattern identifying near-real time products
        (default:nrt_global_allsat_msla_uv_*.nc')

    RETURNS
    -------
    ds : xarray.Dataset
        combined Aviso dataset
    start_date : datetime
        date of first aviso data
    transition_date : datetime
        date when data switches from delayed-time and near-real time
    """
    ds_dt = xr.open_mfdataset(ddir_dt+'/'+fid_dt).sortby('time')
    if ddir_nrt is not None:
        transition_date = ds_dt.time.isel(time=-1)
        ds_nrt = xr.open_mfdataset(ddir_nrt+'/'+fid_nrt).sortby('time')
        ds = xr.concat((ds_dt,
                        ds_nrt.isel(time=ds_nrt.time > transition_date)),
                       dim='time')
    else:
        ds = ds_dt
        transition_date = ''

    # Test if time is continous
    if np.any(ds.time.diff('time').data != ds.time.diff('time')[0].data):
        print(ds.time)
        raise RuntimeError('Time steps are not homogeneous. Likely missing \
        files between the dt and nrt products')

    start_date = ds.time[0].data

    return ds, start_date, transition_date


def interpolated_aviso_validmask(da, xi, yi):
    x = da.lon.data
    y = da.lat.data
    validmask_coarse = ~xr.ufuncs.isnan(da).all(dim='time').data.compute()
    validmask_fine = interp_map_regular_grid(validmask_coarse, x, y, xi, yi)
    return np.isclose(validmask_fine, 1.0)


def block_interpolate(array, x, y, xi, yi):
    a = interp_map_regular_grid(np.squeeze(array), x, y, xi, yi)
    print(a)
    return a[np.newaxis, :, :]


def block_write(array, filename='', block_id=None):
    writebin(array, filename + '%04i' % block_id[0])
    return np.array([1])


def interpolate_write(ds, xi, yi, filename=''):
    x = ds.lon.data
    y = ds.lat.data
    interpolated = ds.data.map_blocks(block_interpolate, x, y, xi, yi,
                                      dtype=np.float64,
                                      chunks=(1, len(yi), len(xi)))
    print('Writing interpolated data to file')
    with ProgressBar():
        interpolated.map_blocks(block_write,
                                filename=filename,
                                dtype=np.float64,
                                chunks=[1], drop_axis=[1, 2]).compute()
    return interpolated


def process_aviso(odir,
                  gdir,
                  ddir_dt,
                  fid_dt='dt_global_allsat_msla_uv_*.nc',
                  ddir_nrt=None,
                  fid_nrt='nrt_global_allsat_msla_uv_*.nc',
                  debug=True):

    """read aviso files into xarray dataset, respecting 'seam' between
    delayed-time
    product and near-real time products

    PARAMETERS
    ----------
    odir : path
        output directory
    gdir : path
        grid directory for the interpolation target
    ddir_dt : path
        data directory for delayed time product
    fid_dt : str
        string pattern identifying delayed time products
        (default:'dt_global_allsat_msla_uv_*.nc')
    ddir_dt : path
        data directory for near-real time product
        (default: None)
    fid_dt : str
        string pattern identifying near-real time products
        (default:nrt_global_allsat_msla_uv_*.nc')
    """
    ds, start_date, transition_date = merge_aviso(ddir_dt,
                                                  fid_dt=fid_dt,
                                                  ddir_nrt=ddir_nrt,
                                                  fid_nrt=fid_nrt)

    # Write out the startdate and transition_date to textfile
    writetxt(str(start_date), odir+'/startdate.txt')
    print('--- startdate written to '+odir+'/startdate.txt ---')

    writetxt(str(transition_date), odir+'/transitiondate.txt.txt')
    print('--- transition date written to '+odir+'/startdate.txt ---')

    grid = open_mdsdataset(gdir, iters=None)
    XC = grid.XC.data
    XG = grid.XG.data
    YC = grid.YC.data
    YG = grid.YG.data

    # create and save validmask
    # validmask indicates values that were interpolated or filled
    # and should be taken out for certain interpretations.
    validmask_aviso_u = interpolated_aviso_validmask(ds.u, XG, YC)
    validmask_aviso_v = interpolated_aviso_validmask(ds.v, XC, YG)

    validmask = np.logical_and(validmask_aviso_u, validmask_aviso_v)

    writebin(validmask, odir+'/validmask.bin')
    print('--- validmask written to '+odir+'/validmask.bin ---')

    #  Velocities near the coast are padded with zeros
    ds = ds.fillna(0)

    # interpolate velocities
    u_interpolated = interpolate_write(ds.u, XG, YC, filename=odir+'/uvel')

    v_interpolated = interpolate_write(ds.v, XC, YG, filename=odir+'/vvel')

    if xr.ufuncs.isnan(u_interpolated).any().compute():
        raise RuntimeError('Nans detected in the u fields')
    if xr.ufuncs.isnan(v_interpolated).any().compute():
        raise RuntimeError('Nans detected in the v fields')


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
