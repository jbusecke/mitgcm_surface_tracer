from __future__ import print_function
import pytest
import xarray as xr
import numpy as np
from mitgcm_surface_tracer.velocity_processing import \
    (aviso_validmask, interpolate_aviso)
from mitgcm_surface_tracer.utils import writebin, readbin
from numpy.testing import assert_allclose
# from xarray.testing import assert_allclose as xr_assert_allclose
import pandas as pd


@pytest.fixture(scope='session')
def vel_dir(tmpdir_factory):
    # Create test arrays
    u_dt = np.array([
                [[np.nan, 1.0],
                 [1.0, 1.0]],
                [[np.nan, 2.0],
                 [2.0, 2.0]],
                [[np.nan, 3.0],
                 [3.0, 3.0]]
                ])

    v_dt = -np.array([
                    [[np.nan, 1.0],
                     [1.0, 1.0]],
                    [[np.nan, 2.0],
                     [2.0, 2.0]],
                    [[np.nan, 3.0],
                     [3.0, 3.0]]
                    ])

    u_nrt = np.array([
                    [[np.nan, 10.0],
                     [10.0, 10.0]],
                    [[np.nan, 20.0],
                     [20.0, 20.0]]
                    ])

    v_nrt = -np.array([
                    [[np.nan, 10.0],
                     [10.0, 10.0]],
                    [[np.nan, 20.0],
                     [20.0, 20.0]]
                    ])
    vel_dir = tmpdir_factory.mktemp('vel_test')

    lon = range(0, 2)
    lat = range(200, 202)

    for tt in range(u_dt.shape[0]):
        file = str(vel_dir.join('dt_'+str(tt)+'.nc'))
        u = u_dt[tt, :, :]
        v = v_dt[tt, :, :]
        time = pd.date_range('2000-01-%02i' % (u_dt.shape[0]-tt), periods=1)
        xr.Dataset({'u':
                    xr.DataArray(u[np.newaxis, :, :],
                                 dims=['time', 'lon', 'lat'],
                                 coords=[time, lon, lat]),
                    'v':
                    xr.DataArray(v[np.newaxis, :, :],
                                 dims=['time', 'lon', 'lat'],
                                 coords=[time, lon, lat]),
                    }).to_netcdf(file)

    for tt in range(u_dt.shape[0]):
        file = str(vel_dir.join('missing_dt_'+str(tt)+'.nc'))
        u = u_dt[tt, :, :]
        v = v_dt[tt, :, :]
        # introduce a 'gap'
        if tt != 0:
            tt = tt + 1
        time = pd.date_range('2000-01-%02i' % (tt+1), periods=1)
        xr.Dataset({'u':
                    xr.DataArray(u[np.newaxis, :, :],
                                 dims=['time', 'lon', 'lat'],
                                 coords=[time, lon, lat]),
                    'v':
                    xr.DataArray(v[np.newaxis, :, :],
                                 dims=['time', 'lon', 'lat'],
                                 coords=[time, lon, lat]),
                    }).to_netcdf(file)

    for tt in range(u_nrt.shape[0]):
        file = str(vel_dir.join('nrt_'+str(tt)+'.nc'))
        u = u_nrt[tt, :, :]
        v = v_nrt[tt, :, :]
        time = pd.date_range('2000-01-%02i' % (tt+3), periods=1)
        xr.Dataset({'u':
                    xr.DataArray(u[np.newaxis, :, :],
                                 dims=['time', 'lon', 'lat'],
                                 coords=[time, lon, lat]),
                    'v':
                    xr.DataArray(v[np.newaxis, :, :],
                                 dims=['time', 'lon', 'lat'],
                                 coords=[time, lon, lat]),
                    }).to_netcdf(file)

    for tt in range(u_nrt.shape[0]):
        file = str(vel_dir.join('missing_nrt_'+str(tt)+'.nc'))
        u = u_nrt[tt, :, :]
        v = v_nrt[tt, :, :]
        time = pd.date_range('2000-01-%02i' % (tt+5), periods=1)
        xr.Dataset({'u':
                    xr.DataArray(u[np.newaxis, :, :],
                                 dims=['time', 'lon', 'lat'],
                                 coords=[time, lon, lat]),
                    'v':
                    xr.DataArray(v[np.newaxis, :, :],
                                 dims=['time', 'lon', 'lat'],
                                 coords=[time, lon, lat]),
                    }).to_netcdf(file)
    return vel_dir


def test_aviso_validmask(vel_dir):
    ds = xr.open_mfdataset(str(vel_dir.join('dt_*.nc')))
    data = ds.u
    xi = np.linspace(data.lon.data.min(), data.lon.data.max(), 3)
    yi = np.linspace(data.lat.data.min(), data.lat.data.max(), 3)
    mask = aviso_validmask(data, xi, yi)
    mask_expected = np.array([
                             [False, False, True],
                             [False, False, True],
                             [True, True, True]
                             ])
    assert_allclose(mask_expected, mask)


def test_interpolate_aviso(vel_dir):
    ds = xr.open_mfdataset(str(vel_dir.join('dt_*.nc')))
    ds = ds.sortby('time').chunk({'time': 1})
    data = ds.u
    xc = np.linspace(data.lon.data.min(), data.lon.data.max(), 3)
    yc = np.linspace(data.lat.data.min(), data.lat.data.max(), 4)
    xg = np.linspace(data.lon.data.min(), data.lon.data.max(), 3)
    yg = np.linspace(data.lat.data.min(), data.lat.data.max(), 4)

    ds_int, validmask = interpolate_aviso(ds, XG=xg, XC=xc, YG=yg, YC=yc)
    print(ds.isel(lat=1, lon=1).u)
    u_control = np.array([
        [[0., 1.5, 3.],
         [1., 2., 3.],
         [2., 2.5, 3.],
         [3., 3., 3.]],

        [[0., 1., 2.],
         [0.66666667, 1.33333333, 2.],
         [1.33333333, 1.66666667, 2.],
         [2., 2., 2.]],

        [[0., 0.5, 1.],
         [0.33333333, 0.66666667, 1.],
         [0.66666667,  0.83333333,  1.],
         [1., 1., 1.]]
        ])

    v_control = np.array([
        [[0., 1.5, 3.],
         [1., 2., 3.],
         [2., 2.5, 3.],
         [3., 3., 3.]],

        [[0., 1., 2.],
         [0.66666667, 1.33333333, 2.],
         [1.33333333, 1.66666667, 2.],
         [2., 2., 2.]],

        [[0., 0.5, 1.],
         [0.33333333, 0.66666667, 1.],
         [0.66666667,  0.83333333,  1.],
         [1., 1., 1.]]
        ])
    print('pooo')
    print(ds_int.u.shape)
    print(u_control.shape)
    print('pooo')
    assert_allclose(u_control, ds_int.u.compute())
    # TODO:Not sure what I would expect here...need a lon wrapped dataset and
    # check if the values are
    # assert_allclose(v_control, v.compute())


# def test_combine_validmask(tmpdir):
#     a = np.array([
#                  [True, True, True],
#                  [False, True, False]
#                  ])
#
#     b = np.array([
#                  [False, True, True],
#                  [True, False, False]
#                  ])
#
#     c = np.array([
#                  [False, True, True],
#                  [True, True, False]
#                  ])
#
#     combo = np.array([
#                      [False, True, True],
#                      [False, False, False]
#                      ])
#
#     file = tmpdir.mkdir('a').join('validmask.bin')
#     writebin(a, file.strpath)
#     file = tmpdir.mkdir('b').join('validmask.bin')
#     writebin(b, file.strpath)
#     file = tmpdir.mkdir('c').join('validmask.bin')
#     writebin(c, file.strpath)
#
#     dir = tmpdir.strpath
#
#     with pytest.raises(RuntimeWarning):
#         combine_validmask(dir, shape=None)
#
#     combine_validmask(dir, a.shape, debug=True)
#
#     test_file = tmpdir.join('validmask_combined.bin')
#     assert_allclose(combo, readbin(test_file.strpath, a.shape))
