# from __future__ import print_function
# from future.utils import iteritems
# import pytest
# import xarray as xr
import numpy as np
import os

from mitgcm_surface_tracer.utils import writebin, readbin, dirCheck


def test_bin_io():
    # This could probably be written nicer with tmpdir
    filename = 'test_bin_io_file.bin'
    a = np.random.random(size=[10, 20])
    writebin(a, filename)
    b = readbin(filename, a.shape)
    assert np.isclose(a, b).all()
    os.remove(filename)


def test_dirCheck():
    testdir = 'testdir'
    if os.path.exists(testdir):
        os.rmdir(testdir)
    checkdir = dirCheck(testdir, True)
    assert checkdir == testdir+'/'
    assert os.path.exists(testdir)
    os.rmdir(testdir)
    checkdir = dirCheck(testdir, False)
    assert checkdir == testdir+'/'
    assert ~os.path.exists(testdir)
