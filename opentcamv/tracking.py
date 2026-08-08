"""Per-tid0 forward/backward tracking, via a time-window slice of the input data.

`pyvttrac.track()` re-scans/re-copies its entire `z`/`mask` on every call,
which is cheap once but ruinous per-tid0 over hundreds of frames. Slicing
the minimal `[tid0 - back, tid0 + fwd]` window before calling keeps that
cost independent of the full data's `nt`, and confines `--revrot` rotation
to exactly the frames a given tid0's tracking will touch.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .params import TrackingSetup


def window_bounds(tid0: int, setup: TrackingSetup, nt: int) -> "tuple[int, int]":
    """`(lo, hi)`: inclusive index bounds of the time window a trajectory
    starting at `tid0` needs. Guaranteed in-range by how `tg` (the set of
    valid `tid0` values) is constructed."""
    half_bwd = setup.nsteps * setup.itstep if setup.backward else 0
    half_fwd = setup.nsteps * setup.itstep if setup.forward else 0
    lo, hi = tid0 - half_bwd, tid0 + half_fwd
    assert 0 <= lo and hi <= nt - 1, f"window [{lo}, {hi}] out of range for nt={nt} (tid0={tid0})"
    return lo, hi


def _globalize_t_index(t_index: np.ndarray, lo: int) -> np.ndarray:
    return np.where(t_index < 0, -1, t_index + lo).astype(np.int64)


def track_at(
    tracker,
    setup: TrackingSetup,
    z_win: np.ndarray,
    t_win: np.ndarray,
    mask_win: "Optional[np.ndarray]",
    grid,
    t0_local: int,
    lo: int,
    xxg: np.ndarray,
    yyg: np.ndarray,
    diagnostics=False,
):
    """Forward and/or backward tracking for one `tid0`, from an
    already-sliced (and, if `--revrot` is active, already-rotated) time
    window `z_win`/`t_win`/`mask_win`. `t0_local` is `tid0`'s index within
    the window; `lo` is the window's global start index, used to translate
    `t_index` back to a global index (`t_index + lo`, except `-1` stays
    `-1`).

    Returns `(fwd, bwd)`; the direction(s) not requested by `setup.forward`/
    `setup.backward` come back as `None`.
    """
    fwd = bwd = None
    if setup.forward:
        fwd = tracker.track(
            z_win, xxg, yyg, t0=t0_local, step=setup.itstep,
            time=t_win, mask=mask_win, grid=grid, diagnostics=diagnostics,
        )
        fwd.t_index = _globalize_t_index(fwd.t_index, lo)
    if setup.backward:
        bwd = tracker.track(
            z_win, xxg, yyg, t0=t0_local, step=-setup.itstep,
            time=t_win, mask=mask_win, grid=grid, diagnostics=diagnostics,
        )
        bwd.t_index = _globalize_t_index(bwd.t_index, lo)
    return fwd, bwd


def diagnostics_arg(args):
    """v1's independent `--out_subimage`/`--out_score_ary` flags, as a
    `pyvttrac.track(diagnostics=...)` value."""
    parts = []
    if args.out_subimage:
        parts.append("templates")
    if args.out_score_ary:
        parts.append("score_grids")
    return tuple(parts) if parts else False
