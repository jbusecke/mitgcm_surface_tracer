from __future__ import print_function
import pytest
import xarray as xr
import numpy as np
from mitgcm_surface_tracer.velocity_processing import (validmask_aviso,
                                                       combine_validmask)
from mitgcm_surface_tracer.utils import writebin, readbin
from numpy.testing import assert_allclose


def test_validmask_aviso():
    time = np.array([0, 1, 2, 3])
    x = np.array([0, 1, 2])
    data1 = np.array([
                     [1, np.nan, 1, 1],
                     [np.nan, 1, 1, np.nan],
                     [1, 1, 1, 1]
                     ]
                     )

    data2 = np.array([
                     [1, np.nan, 1, 1],
                     [np.nan, 1, np.nan, np.nan],
                     [1, 1, 1, np.nan]
                     ]
                     )

    a = xr.DataArray(data1,
                     coords=[x, time],
                     dims=['x', 'time'])

    b = xr.DataArray(data2,
                     coords=[x, time],
                     dims=['x', 'time'])

    mask = validmask_aviso(a, a)
    test = np.all(~np.isnan(a), axis=a.get_axis_num('time'))

    assert mask.data == test.data

    with pytest.raises(RuntimeWarning):
        validmask_aviso(a, b)


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
