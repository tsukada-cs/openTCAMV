"""rotate_window: rotation angle/direction, NaN fill, and that every frame
in a window is rotated (not just the ones landing on the it_rel grid)."""

from __future__ import annotations

import numpy as np

from opentcamv.rotation import rotate_frame, rotate_mask_window, rotate_window, spline_coefficients


def test_reference_frame_i0_is_untouched():
    rng = np.random.RandomState(0)
    z = rng.rand(3, 21, 21).astype(np.float32)
    t = np.array([0.0, 10.0, 20.0])
    out = rotate_window(z, t, i0=1, omega=0.05)
    np.testing.assert_array_equal(out[1], z[1])


def test_zero_omega_is_identity():
    rng = np.random.RandomState(1)
    z = rng.rand(4, 21, 21).astype(np.float32)
    t = np.array([0.0, 5.0, 10.0, 15.0])
    out = rotate_window(z, t, i0=0, omega=0.0)
    np.testing.assert_allclose(out, z)


def test_positive_angle_rotates_counter_clockwise_in_array_space():
    # A single bright pixel above-center; rotating 90 deg CCW (PIL
    # convention) should move it to the left of center.
    n = 41
    c = n // 2
    z = np.zeros((n, n), dtype=np.float32)
    z[c - 10, c] = 1.0  # "up" (lower row index) from center
    z2 = np.stack([z, z])
    t = np.array([0.0, 1.0])
    # deg_per_sec = rad2deg(omega); want a 90 deg rotation over dt=1
    omega = np.deg2rad(90.0)
    out = rotate_window(z2, t, i0=0, omega=omega)
    peak = np.unravel_index(np.argmax(out[1]), out[1].shape)
    assert peak == (c, c - 10)  # moved to "left" of center: CCW


def test_nan_fill_propagates_and_stays_bounded_to_corners():
    rng = np.random.RandomState(2)
    z = (50 + 10 * rng.rand(2, 31, 31)).astype(np.float32)
    t = np.array([0.0, 20.0])
    out = rotate_window(z, t, i0=0, omega=np.deg2rad(20.0))
    assert np.isnan(out[1]).any()
    assert not np.isnan(out[1]).all()
    # Center pixel, far from the rotated-in corners, should stay valid.
    assert not np.isnan(out[1][15, 15])


def test_all_frames_in_window_are_rotated_not_just_it_rel_spaced():
    # With traj_int > 1, v1 only rotated it_rel-spaced frames; window-based
    # rotation must rotate every frame regardless.
    rng = np.random.RandomState(3)
    z = rng.rand(5, 25, 25).astype(np.float32)
    t = np.arange(5, dtype=np.float64) * 10.0
    omega = np.deg2rad(15.0)
    out = rotate_window(z, t, i0=0, omega=omega)
    for k in (1, 2, 3, 4):  # not just k=2,4 (an "it_rel spaced by 2" subset)
        coefficients, valid = spline_coefficients(z[k])
        expected = rotate_frame(coefficients, valid, omega * t[k])
        np.testing.assert_allclose(out[k], expected, equal_nan=True, atol=1e-5)


def test_rotation_is_reversible_to_spline_accuracy():
    # The reason for using an interpolating spline rather than PIL's
    # smoothing "bicubic": a rotate-and-rotate-back round trip has to give
    # the field back, or every rotated frame is a blurred one.
    n = 61
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64) - c
    f = (np.sin(2 * np.pi * xx / 25.0) * np.cos(2 * np.pi * yy / 31.0) * 30 + 250).astype(np.float32)
    t = np.array([0.0, 1.0])
    theta = np.deg2rad(7.3)
    there = rotate_window(np.stack([f, f]), t, i0=0, omega=theta)[1]
    back = rotate_window(np.stack([there, there]), t, i0=0, omega=-theta)[1]
    interior = (np.hypot(xx, yy) < n / 4) & np.isfinite(back)
    assert np.abs(back - f)[interior].max() < 0.05


def test_mask_rotates_with_the_same_angles_and_stays_boolean():
    n = 41
    c = n // 2
    mask = np.zeros((2, n, n), dtype=bool)
    mask[:, c - 10, c] = True  # "up" (lower row index) from center
    t = np.array([0.0, 1.0])
    out = rotate_mask_window(mask, t, i0=0, omega=np.deg2rad(90.0))
    assert out.dtype == np.bool_
    np.testing.assert_array_equal(out[0], mask[0])  # reference frame untouched
    assert out[1][c, c - 10]  # same CCW quarter turn rotate_window applies
    assert out[1].sum() == 1


def test_rotated_in_corners_are_masked_out():
    mask = np.zeros((2, 31, 31), dtype=bool)
    t = np.array([0.0, 1.0])
    out = rotate_mask_window(mask, t, i0=0, omega=np.deg2rad(20.0))
    assert out[1][0, 0]  # nothing is known about a corner pulled in from outside
    assert not out[1][15, 15]


def test_float64_input_is_not_silently_corrupted():
    # PIL's mode="F" reinterprets a float64 buffer as float32 and returns
    # garbage without raising; nothing in this module may do that.
    z = np.stack([np.arange(25, dtype=np.float64).reshape(5, 5) + 100.0] * 2)
    t = np.array([0.0, 1.0])
    out = rotate_window(z, t, i0=0, omega=0.0)
    np.testing.assert_allclose(out[1], z[1])


def test_out_buffer_is_used_when_provided():
    rng = np.random.RandomState(4)
    z = rng.rand(2, 15, 15).astype(np.float32)
    t = np.array([0.0, 5.0])
    buf = np.empty_like(z)
    out = rotate_window(z, t, i0=0, omega=0.01, out=buf)
    assert out is buf
