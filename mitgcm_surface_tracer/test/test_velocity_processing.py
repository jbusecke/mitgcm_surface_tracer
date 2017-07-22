from __future__ import print_function
import pytest
import xarray as xr
import numpy as np
from mitgcm_surface_tracer.velocity_processing import \
    (interpolated_aviso_validmask, combine_validmask, merge_aviso,
     interpolate_write)
from mitgcm_surface_tracer.utils import writebin, readbin
from numpy.testing import assert_allclose
import pandas as pd


@pytest.fixture(scope='session')
def vel_dir(tmpdir_factory):
    # Create test arrays
    u_rt = np.array([
                [[np.nan, 1.0],
                 [1.0, 1.0]],
                [[np.nan, 2.0],
                 [2.0, 2.0]],
                [[np.nan, 3.0],
                 [3.0, 3.0]]
                ])

    v_rt = -np.array([
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

    for tt in range(u_rt.shape[0]):
        file = str(vel_dir.join('rt_'+str(tt)+'.nc'))
        u = u_rt[tt, :, :]
        v = v_rt[tt, :, :]
        time = pd.date_range('2000-01-%02i' % (u_rt.shape[0]-tt), periods=1)
        xr.Dataset({'u':
                    xr.DataArray(u[np.newaxis, :, :],
                                 dims=['time', 'lon', 'lat'],
                                 coords=[time, lon, lat]),
                    'v':
                    xr.DataArray(v[np.newaxis, :, :],
                                 dims=['time', 'lon', 'lat'],
                                 coords=[time, lon, lat]),
                    }).to_netcdf(file)

    for tt in range(u_rt.shape[0]):
        file = str(vel_dir.join('missing_rt_'+str(tt)+'.nc'))
        u = u_rt[tt, :, :]
        v = v_rt[tt, :, :]
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
    with pytest.raises(RuntimeError) as excinfo:
        merge_aviso(ddir,
                    fid_dt='rt*.nc',
                    ddir_nrt=ddir,
                    fid_nrt='missing_nrt_*.nc')
        assert 'Time steps are not homogeneous' in excinfo.value.message

    with pytest.raises(RuntimeError) as excinfo:
        merge_aviso(ddir,
                    fid_dt='missing_rt*.nc',
                    ddir_nrt=ddir,
                    fid_nrt='nrt_*.nc')


def test_interpolated_aviso_validmask(vel_dir):
    ds = xr.open_mfdataset(str(vel_dir.join('rt_*.nc')))
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
    ds = xr.open_mfdataset(str(vel_dir.join('rt_*.nc')))
    data = ds.u
    xi = np.linspace(data.lon.data.min(), data.lon.data.max(), 3)
    yi = np.linspace(data.lat.data.min(), data.lat.data.max(), 4)

    fname = str(vel_dir)+'/write_test'
    interpolated = interpolate_write(data, xi, yi, filename=fname)

    fname_padded = str(vel_dir)+'/write_test_padded'
    interpolated_padded = interpolate_write(data.fillna(0), xi, yi,
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
