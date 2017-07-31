from __future__ import print_function
import xarray as xr
import numpy as np
# import time
import os
import os.path
from xmitgcm import open_mdsdataset
from xarrayutils.numpy_utils import interp_map_regular_grid
from .utils import readbin, writebin, writetxt, writable_mds_store
from dask.diagnostics import ProgressBar
from aviso_products.aviso_processing import merge_aviso
=======
from .utils import readbin, writebin, writetxt
from dask.diagnostics import ProgressBar


def interpolated_aviso_validmask(da, xi, yi):
    x = da.lon.data
    y = da.lat.data
    validmask_coarse = ~xr.ufuncs.isnan(da).all(dim='time').data.compute()
    validmask_fine = interp_map_regular_grid(validmask_coarse, x, y, xi, yi)
    return np.isclose(validmask_fine, 1.0)


def block_interpolate(array, x, y, xi, yi):
    a = interp_map_regular_grid(np.squeeze(array), x, y, xi, yi)
    return a[np.newaxis, :, :]

def interpolated_aviso_validmask(da, xi, yi):
    x = da.lon.data
    y = da.lat.data
    validmask_coarse = ~xr.ufuncs.isnan(da).all(dim='time').data.compute()
    validmask_fine = interp_map_regular_grid(validmask_coarse, x, y, xi, yi)
    return np.isclose(validmask_fine, 1.0)
  

def process_aviso(odir,
                  ddir_dt,
                  xc=None,
                  xg=None,
                  yc=None,
                  yg=None,
                  fid_dt='dt_global_allsat_msla_uv_*.nc',
                  gdir=None,
                  ddir_nrt=None,
                  fid_nrt='nrt_global_allsat_msla_uv_*.nc',
                  debug=True,
                  verbose=True,
                  mkdir=False):

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
    if mkdir:
        if not os.path.exists(odir):
            os.mkdir(odir)

    if gdir is None:
        if any([x is None for x in [xg, yg, xc, yc]]):
            raise RuntimeError('if grid dir is not specified all interpolation\
            coordinates have to be supplied as input')
        XC = xc
        XG = xg
        YC = yc
        YG = yg
    else:
        if any([x is not None for x in [xg, yg, xc, yc]]):
            raise RuntimeError('if grid dir is supplied, interpolation\
             coordinates can not be specified')

        grid = open_mdsdataset(gdir, iters=None)
        XC = grid.XC.data
        XG = grid.XG.data
        YC = grid.YC.data
        YG = grid.YG.data

    ds, start_date, transition_date = merge_aviso(ddir_dt,
                                                  fid_dt=fid_dt,
                                                  ddir_nrt=ddir_nrt,
                                                  fid_nrt=fid_nrt)
    if verbose:
        print('Startdate:'+str(start_date))
    writetxt(str(start_date), odir+'/startdate.txt', verbose=verbose)

    if verbose:
        print('Near-real-time Transition:'+str(transition_date))
    writetxt(str(transition_date), odir+'/transitiondate.txt', verbose=verbose)

    # create and save validmask
    # validmask indicates values that were interpolated or filled
    # and should be taken out for certain interpretations.
    validmask_aviso_u = interpolated_aviso_validmask(ds.u, XG, YC)
    validmask_aviso_v = interpolated_aviso_validmask(ds.v, XC, YG)
    validmask = np.logical_and(validmask_aviso_u, validmask_aviso_v)

    if verbose:
        print ('Validmask')
    writebin(validmask, odir+'/validmask.bin', verbose=verbose)

    #  Velocities near the coast are padded with zeros and then interpolated
    ds = ds.fillna(0)

    x = ds.lon.data
    y = ds.lat.data

    u_interpolated = ds.u.data.map_blocks(block_interpolate, x, y, XG, YC,
                                          dtype=np.float64,
                                          chunks=(1, len(YC), len(XG)))

    v_interpolated = ds.v.data.map_blocks(block_interpolate, x, y, XC, YG,
                                          dtype=np.float64,
                                          chunks=(1, len(YG), len(XC)))
    iters = range(len(ds.time.data))
    uvel_store = writable_mds_store(os.path.join(odir, 'uvelCorr'), iters)
    vvel_store = writable_mds_store(os.path.join(odir, 'vvelCorr'), iters)

    if verbose:
        print('Writing interpolated u velocities to ' + odir + 'uvel')
    with ProgressBar():
        u_interpolated.store(uvel_store)

    if verbose:
        print('Writing interpolated v velocities to ' + odir + 'vvel')
    with ProgressBar():
        v_interpolated.store(vvel_store)

    return u_interpolated, v_interpolated


def combine_validmask(data_dir, shape=None, debug=False):
    fnames = []
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for filename in [f for f in filenames if f == 'validmask.bin']:
            print('found validmask at '+os.path.join(dirpath, filename))
            fnames.append(os.path.join(dirpath, filename))
    if debug:
        print('data_dir', data_dir)
        print(fnames)

    fpath = data_dir+'/validmask_combined.bin'
    writebin(combo, fpath)
    print('--- combined validmask written to '+fpath+' ---')
