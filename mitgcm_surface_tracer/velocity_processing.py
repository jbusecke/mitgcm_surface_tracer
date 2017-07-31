from __future__ import print_function
import xarray as xr
import numpy as np
# import time
import os
import os.path
from xarrayutils.numpy_utils import interp_map_regular_grid
from .utils import writable_mds_store
from dask.diagnostics import ProgressBar


def aviso_validmask(da, xi, yi):
    x = da.lon.data
    y = da.lat.data
    validmask_coarse = ~xr.ufuncs.isnan(da).all(dim='time').data.compute()
    validmask_fine = interp_map_regular_grid(validmask_coarse, x, y, xi, yi)
    return np.isclose(validmask_fine, 1.0)


def block_interpolate(array, x, y, xi, yi):
    a = interp_map_regular_grid(np.squeeze(array), x, y, xi, yi)
    return a[np.newaxis, :, :]


def interpolate_aviso(ds, XC, XG, YC, YG,
                      debug=True, verbose=True, mkdir=False):

    """Interpolate aviso dataset onto model coordinates (regular lat lon grid)

    PARAMETERS
    ----------
    ds : xarray Dataset from reading in Aviso data
        (e.g. from aviso_products.merge_aviso)

    XC : numpy.array Longitude at cell center (needs to be a vector)

    XG : numpy.array Longitude at cell boundary

    YC : numpy.array Latitude at cell center

    YG : numpy.array Latitude at cell boundary

    RETURNS
    -------
    ds_interpolated : xarray Dataset with interpolated values
    validmask : indicates data points that were not interpolated
    """
    # if mkdir:
    #     if not os.path.exists(odir):
    #         os.mkdir(odir)

    # create and save validmask
    # validmask indicates values that were interpolated or filled
    # and should be taken out for certain interpretations.
    validmask_aviso_u = aviso_validmask(ds.u, XG, YC)
    validmask_aviso_v = aviso_validmask(ds.v, XC, YG)
    validmask = np.logical_and(validmask_aviso_u, validmask_aviso_v)

    # if verbose:
    #     print ('Validmask')
    # writebin(validmask, odir+'/validmask.bin', verbose=verbose)

    #  Velocities near the coast are padded with zeros and then interpolated
    ds = ds.fillna(0)

    x = ds.lon.data
    y = ds.lat.data

    u_interpolated = ds.u.data.map_blocks(block_interpolate, x, y, XG, YC,
                                          dtype=np.float64,
                                          chunks=(1, len(YC), len(XG)))

    u_interpolated = xr.DataArray(u_interpolated,
                                  dims=['time', 'lon', 'lat'],
                                  coords={'time': ds.time,
                                          'lon': XG,
                                          'lat': YC
                                          })

    v_interpolated = ds.v.data.map_blocks(block_interpolate, x, y, XC, YG,
                                          dtype=np.float64,
                                          chunks=(1, len(YG), len(XC)))

    v_interpolated = xr.DataArray(v_interpolated,
                                  dims=['time', 'lon', 'lat'],
                                  coords={'time': ds.time,
                                          'lon': XG,
                                          'lat': YC
                                          })

    ds_interpolated = xr.Dataset({'u': u_interpolated,
                                  'v': v_interpolated})

    return ds_interpolated, validmask


def aviso_store_daily(ds, odir, verbose=True):
    iters = range(len(ds.time.data))
    uvel_store = writable_mds_store(os.path.join(odir, 'uvelCorr'), iters)
    vvel_store = writable_mds_store(os.path.join(odir, 'vvelCorr'), iters)

    if verbose:
        print('Writing interpolated u velocities to ' + odir + 'uvel')
    with ProgressBar():
        ds.u.data.store(uvel_store)

    if verbose:
        print('Writing interpolated v velocities to ' + odir + 'vvel')
    with ProgressBar():
        ds.v.data.store(vvel_store)


# def combine_validmask(data_dir, shape=None, debug=False):
#     fnames = []
#     for dirpath, dirnames, filenames in os.walk(data_dir):
#         for filename in [f for f in filenames if f == 'validmask.bin']:
#             print('found validmask at '+os.path.join(dirpath, filename))
#             fnames.append(os.path.join(dirpath, filename))
#     if debug:
#         print('data_dir', data_dir)
#         print(fnames)
#
#     if shape:
#         masks = np.array([readbin(f, shape) for f in fnames])
#     else:
#         raise RuntimeWarning('When shape is not given')
#
#     combo = np.all(np.stack(masks, axis=2), axis=2)
#
#     fpath = data_dir+'/validmask_combined.bin'
#     writebin(combo, fpath)
#     print('--- combined validmask written to '+fpath+' ---')
