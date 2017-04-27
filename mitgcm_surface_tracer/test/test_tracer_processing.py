import numpy as np
from mitgcm_surface_tracer.tracer_processing import reset_cut
from numpy.testing import assert_allclose


def test_reset_cut():
    # Case with no phase
    n = 5
    dt_model = 2
    dt_tracer = 2
    iters = np.array(range(n))
    reset_frq = 4
    cut_time = 2
    reset_pha = 0
    mask, reset_iters, reset_time = reset_cut(reset_frq, reset_pha,
                                              dt_model, dt_tracer,
                                              iters, cut_time)
    assert_allclose(reset_iters, np.array([0, 2, 4]))
    assert_allclose(reset_time, np.array([0, 4, 8]))
    assert_allclose(mask, np.array([False, True, False, True, False]))
    assert mask.dtype == bool

    # Case with phase on time step
    reset_pha = 2
    mask, reset_iters, reset_time = reset_cut(reset_frq, reset_pha,
                                              dt_model, dt_tracer,
                                              iters, cut_time)
    assert_allclose(reset_iters, np.array([0, 1, 3]))
    assert_allclose(reset_time, np.array([0, 2, 6]))
    assert_allclose(mask, np.array([False, False, True, False, True]))
    assert mask.dtype == bool

    # Case with phase off time step
    n = 6
    iters = np.array(range(n))
    reset_frq = 5
    reset_pha = 0
    mask, reset_iters, reset_time = reset_cut(reset_frq, reset_pha,
                                              dt_model, dt_tracer,
                                              iters, cut_time)
    assert_allclose(reset_iters, np.array([0, 3, 5]))
    assert_allclose(reset_time, np.array([0, 5, 10]))
    assert_allclose(mask, np.array([False, True, True, False,
                                    True, False]))
    assert mask.dtype == bool

    # Case with unequal dt_model dt_tracer
    n = 6
    dt_model = 2
    dt_tracer = 4
    iters = np.array(range(n))
    reset_frq = 5
    cut_time = 4
    reset_pha = 0
    mask, reset_iters, reset_time = reset_cut(reset_frq, reset_pha,
                                              dt_model, dt_tracer,
                                              iters, cut_time)
    assert_allclose(reset_iters, np.array([0, 4]))
    assert_allclose(reset_time, np.array([0, 5, 10]))
    assert_allclose(mask, np.array([False, False, True, True,
                                    False, False]))
    assert mask.dtype == bool

    # Case without reset_cut
    n = 6
    dt_model = 2
    dt_tracer = 4
    iters = np.array(range(n))
    reset_frq = 0
    cut_time = 0
    reset_pha = 0
    mask, reset_iters, reset_time = reset_cut(reset_frq, reset_pha,
                                              dt_model, dt_tracer,
                                              iters, cut_time)
    assert_allclose(reset_iters, np.array([0]))
    assert_allclose(reset_time, np.array([0]))
    assert_allclose(mask, np.array([True, True, True, True,
                                    True, True]))
    assert mask.dtype == bool
