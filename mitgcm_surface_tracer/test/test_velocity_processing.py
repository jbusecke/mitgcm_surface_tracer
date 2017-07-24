from __future__ import print_function
import pytest
import xarray as xr
import numpy as np
from mitgcm_surface_tracer.velocity_processing import \
    (interpolated_aviso_validmask, combine_validmask, merge_aviso,
     interpolate_write, process_aviso)
from mitgcm_surface_tracer.utils import writebin, readbin
from numpy.testing import assert_allclose
from xarray.testing import assert_allclose as xr_assert_allclose
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


def test_merge_aviso(vel_dir):
    ddir = str(vel_dir)

    ds_dt, sd_dt, td_dt = merge_aviso(ddir,
                                      fid_dt='dt*.nc',
                                      ddir_nrt=None)
    ds_dt_check = xr.open_mfdataset(ddir+'/dt*.nc').sortby('time')
    xr_assert_allclose(ds_dt, ds_dt_check)
    assert sd_dt == np.datetime64('2000-01-01')
    assert td_dt is None
    print(ds_dt.chunks)
    assert all([x == 1 for x in list(ds_dt.chunks['time'])])

    ds_nrt, sd_nrt, td_nrt = merge_aviso(ddir,
                                         fid_dt='dt*.nc',
                                         ddir_nrt=ddir,
                                         fid_nrt='nrt_*.nc')
    check_time = slice('2000-01-04', None)
    ds_nrt_check = xr.merge([xr.open_mfdataset(ddir+'/dt*.nc'),
                            xr.open_mfdataset(ddir+'/nrt*.nc').
                            sel(time=check_time)]).sortby('time')

    xr_assert_allclose(ds_nrt, ds_nrt_check)
    assert sd_nrt == np.datetime64('2000-01-01')
    assert td_nrt == np.datetime64('2000-01-03')

    with pytest.raises(RuntimeError) as excinfo:
        merge_aviso(ddir,
                    fid_dt='dt*.nc',
                    ddir_nrt=ddir,
                    fid_nrt='missing_nrt_*.nc')
        assert 'Time steps are not homogeneous' in excinfo.value.message

    with pytest.raises(RuntimeError) as excinfo:
        merge_aviso(ddir,
                    fid_dt='missing_dt*.nc',
                    ddir_nrt=ddir,
                    fid_nrt='nrt_*.nc')


def test_interpolated_aviso_validmask(vel_dir):
    ds = xr.open_mfdataset(str(vel_dir.join('dt_*.nc')))
    data = ds.u
    xi = np.linspace(data.lon.data.min(), data.lon.data.max(), 3)
    yi = np.linspace(data.lat.data.min(), data.lat.data.max(), 3)
    mask = interpolated_aviso_validmask(data, xi, yi)
    mask_expected = np.array([
                             [False, False, True],
                             [False, False, True],
                             [True, True, True]
                             ])
    assert_allclose(mask_expected, mask)


def test_interpolate_write(vel_dir):
    fname = str(vel_dir)+'/write_test'
    ds = xr.open_mfdataset(str(vel_dir.join('dt_*.nc')))
    data = ds.u
    xi = np.linspace(data.lon.data.min(), data.lon.data.max(), 3)
    yi = np.linspace(data.lat.data.min(), data.lat.data.max(), 4)

    fname = str(vel_dir)+'/write_test'
    interpolate_write(data, xi, yi, filename=fname)

    fname_padded = str(vel_dir)+'/write_test_padded'
    interpolate_write(data.fillna(0), xi, yi,
                      filename=fname_padded)

    interpolated_control = np.array([
        [[0., 0.5, 1.],
         [0.33333333, 0.66666667, 1.],
         [0.66666667,  0.83333333,  1.],
         [1., 1., 1.]],

        [[0., 1., 2.],
         [0.66666667, 1.33333333, 2.],
         [1.33333333, 1.66666667, 2.],
         [2., 2., 2.]],

        [[0., 1.5, 3.],
         [1., 2., 3.],
         [2., 2.5, 3.],
         [3., 3., 3.]]
        ])
    for tt in range(len(data.time)):
        assert_allclose(readbin(fname+'%04i' % tt, [4, 3]),
                        np.ones_like(interpolated_control[tt, :, :])*np.nan)

        assert_allclose(readbin(fname_padded+'%04i' % tt, [4, 3]),
                        interpolated_control[tt, :, :])


def test_process_aviso(vel_dir):
    ds = xr.open_mfdataset(str(vel_dir.join('dt_*.nc')))
    data = ds.u
    data = data.sortby('time').chunk([1, 2, 2])
    xc = np.linspace(data.lon.data.min(), data.lon.data.max(), 3)
    yc = np.linspace(data.lat.data.min(), data.lat.data.max(), 4)
    xg = np.linspace(data.lon.data.min(), data.lon.data.max(), 3)
    yg = np.linspace(data.lat.data.min(), data.lat.data.max(), 4)

    u, v = process_aviso(str(vel_dir),
                         str(vel_dir),
                         xg=xg,
                         xc=xc,
                         yg=yg,
                         yc=yc,
                         fid_dt='dt_*.nc')

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
    # print(interpolated_control)
    assert_allclose(u_control, u.compute())
    # TODO:Not sure what I would expect here...need a lon wrapped dataset and
    # check if the values are
    # assert_allclose(v_control, v.compute())

    with pytest.raises(RuntimeError) as excinfo:
        process_aviso(vel_dir,
                      vel_dir,
                      gdir=vel_dir,
                      xc=xc)
        assert 'if grid dir is supplied,' in excinfo.value.message

    with pytest.raises(RuntimeError) as excinfo:
        process_aviso(vel_dir,
                      vel_dir,
                      gdir=None,
                      xc=xc,
                      yc=yc,
                      xg=xg)
        assert 'if grid dir is not specified' in excinfo.value.message


def test_combine_validmask(tmpdir):
    a = np.array([
                 [True, True, True],
                 [False, True, False]
                 ])

    b = np.array([
                 [False, True, True],
                 [True, False, False]
                 ])

    c = np.array([
                 [False, True, True],
                 [True, True, False]
                 ])

    combo = np.array([
                     [False, True, True],
                     [False, False, False]
                     ])

    file = tmpdir.mkdir('a').join('validmask.bin')
    writebin(a, file.strpath)
    file = tmpdir.mkdir('b').join('validmask.bin')
    writebin(b, file.strpath)
    file = tmpdir.mkdir('c').join('validmask.bin')
    writebin(c, file.strpath)

    dir = tmpdir.strpath

    with pytest.raises(RuntimeWarning):
        combine_validmask(dir, shape=None)

    combine_validmask(dir, a.shape, debug=True)

    test_file = tmpdir.join('validmask_combined.bin')
    assert_allclose(combo, readbin(test_file.strpath, a.shape))
