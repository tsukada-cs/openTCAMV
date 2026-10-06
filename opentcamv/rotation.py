"""`--revrot`: rotate a tracking time-window to follow a rigid rotation.

Rotation is a cubic B-spline interpolation (`scipy.ndimage`) about the array
center, which for the odd-sized, origin-centered grids this package is meant
for is exactly the TC center. Three deliberate changes from v1's
frame-rotation code (the first two a consequence of moving to per-tid0 time
windows, rather than genuine "existing bugs"):

1. Every frame in the window is rotated, not just the ones landing on the
   `it_rel` output grid. v1 only rotated `it_rel`-spaced frames, so with
   `--traj_int > 1` the intermediate frames used internally by multi-step
   tracking were silently left unrotated (unnoticed because `sample.sh`
   uses `traj_int=1`). Windowing makes rotating everything the natural
   choice.
2. Invalid pixels come back as NaN, not a float32 sentinel. v1's BICUBIC
   interpolation mixed a huge sentinel (~1e38) with real data at the
   rotation's boundary ring, producing values that didn't equal `zmiss`
   exactly and so were silently treated as valid.
3. The interpolation is a true interpolating cubic spline rather than
   `PIL.Image.rotate(resample=BICUBIC)`. PIL's "bicubic" is a *smoothing*
   cubic-convolution kernel: rotating a band-limited field by +7.3 deg and
   back leaves an RMS error of 0.24 (in field units, on a field of
   peak-to-peak 60) -- about the same as bilinear -- against 0.0003 for the
   spline. That smoothing attacks exactly the high-frequency texture template
   matching scores on, and it is applied to every frame except the reference
   one. The spline is also ~3x cheaper per frame here once the prefilter is
   hoisted (see `spline_coefficients`). Rotating through PIL additionally
   required a float32 round-trip: `Image.fromarray(arr, mode="F")` on a
   float64 array silently reinterprets the buffer and returns garbage, with
   no error raised.

NaN handling: `scipy.ndimage`'s spline prefilter is an IIR filter, so a single
NaN anywhere poisons the entire frame. Invalid pixels are therefore replaced
by the frame mean before filtering, and validity is carried separately through
a linear-interpolated weight; any output pixel drawing on an invalid input
pixel is set back to NaN. This is marginally tighter than v1's bicubic NaN
bleed (4x4 support vs 2x2), i.e. it rejects slightly less, not more.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

_SPLINE_ORDER = 3
_VALID_TOL = 1e-6
_EDGE_TOL = 1e-6


def spline_coefficients(frame: np.ndarray) -> "tuple[np.ndarray, np.ndarray | None]":
    """`(coefficients, valid)` for one frame: the cubic B-spline coefficients
    `map_coordinates(..., prefilter=False)` expects, plus the boolean validity
    mask (`None` when the frame is fully valid, so callers can skip the
    weight pass entirely).

    Exposed separately because the prefilter is rotation-independent: a caller
    rotating one frame by many angles, or reusing a frame across overlapping
    windows, should filter it once and reuse the coefficients.
    """
    valid = np.isfinite(frame)
    if valid.all():
        return ndimage.spline_filter(frame.astype(np.float64), order=_SPLINE_ORDER), None
    if not valid.any():
        return np.zeros(frame.shape, dtype=np.float64), valid
    filled = np.where(valid, frame, np.nanmean(frame))
    return ndimage.spline_filter(filled.astype(np.float64), order=_SPLINE_ORDER), valid


def rotation_coordinates(shape: "tuple[int, int]", theta: float) -> np.ndarray:
    """`map_coordinates` source coordinates that rotate an image by `theta`
    radians counter-clockwise in array space, about the array center."""
    ny, nx = shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
    yy -= cy
    xx -= cx
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    coords = np.stack([cy + xx * sin_t + yy * cos_t, cx + xx * cos_t - yy * sin_t])
    # A quarter turn maps a square grid exactly onto itself, but in floating
    # point the edge samples land an ulp outside it and come back as NaN.
    # Snap only that round-off back onto the boundary; a coordinate genuinely
    # outside the frame stays outside.
    for axis, size in enumerate((ny, nx)):
        edge = coords[axis]
        np.copyto(edge, 0.0, where=(edge < 0.0) & (edge > -_EDGE_TOL))
        np.copyto(edge, size - 1.0, where=(edge > size - 1.0) & (edge < size - 1.0 + _EDGE_TOL))
    return coords


def rotate_frame(coefficients: np.ndarray, valid: "np.ndarray | None", theta: float) -> np.ndarray:
    """One frame, from `spline_coefficients` output, rotated by `theta`
    radians counter-clockwise in array space about the array center."""
    coords = rotation_coordinates(coefficients.shape, theta)
    out = ndimage.map_coordinates(
        coefficients, coords, order=_SPLINE_ORDER, prefilter=False, mode="constant", cval=np.nan
    )
    if valid is not None:
        weight = ndimage.map_coordinates(
            valid.astype(np.float64), coords, order=1, mode="constant", cval=0.0
        )
        out[weight < 1.0 - _VALID_TOL] = np.nan
    return out


def rotate_window(z_base: np.ndarray, t_win: np.ndarray, i0: int, omega: float, out: "np.ndarray | None" = None) -> np.ndarray:
    """Rotate each frame in `z_base` (nwin, ny, nx) by `omega` (rad/s) times
    its elapsed time from frame `i0`. Frame `i0` itself is left untouched
    (elapsed time 0 -> a 0-degree rotation is a no-op, but we skip the
    interpolation for it entirely).

    Positive `omega` rotates later frames counter-clockwise in array space,
    which is clockwise in a physical (x right, y up) frame -- the same
    convention v1's PIL-based implementation had.
    """
    nwin = z_base.shape[0]
    if out is None:
        out = np.empty_like(z_base)
    for k in range(nwin):
        if k == i0:
            out[k] = z_base[k]
            continue
        coefficients, valid = spline_coefficients(z_base[k])
        out[k] = rotate_frame(coefficients, valid, omega * (t_win[k] - t_win[i0]))
    return out


def rotate_mask_window(mask_win: np.ndarray, t_win: np.ndarray, i0: int, omega: float) -> np.ndarray:
    """The `mask` counterpart of `rotate_window` (True = ignore), rotated by
    the same per-frame angles with nearest-neighbor sampling so mask values
    stay exactly True/False. Pixels rotated in from outside the frame are
    True, since nothing is known about them.

    Tracking a rotated `z` against an unrotated `mask` silently scores the
    wrong pixels -- which is what happened before this existed, unnoticed
    because `sample.sh` never sets `--maskvar`.
    """
    nwin = mask_win.shape[0]
    out = np.empty_like(mask_win, dtype=bool)
    for k in range(nwin):
        if k == i0:
            out[k] = mask_win[k]
            continue
        coords = rotation_coordinates(mask_win.shape[1:], omega * (t_win[k] - t_win[i0]))
        out[k] = ndimage.map_coordinates(
            mask_win[k].astype(np.uint8), coords, order=0, mode="constant", cval=1
        ).astype(bool)
    return out
