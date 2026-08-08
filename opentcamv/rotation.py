"""`--revrot`: rotate a tracking time-window to follow a rigid rotation.

Two deliberate changes from v1's frame-rotation code (both a consequence of
moving to per-tid0 time windows, rather than genuine "existing bugs"):

1. Every frame in the window is rotated, not just the ones landing on the
   `it_rel` output grid. v1 only rotated `it_rel`-spaced frames, so with
   `--traj_int > 1` the intermediate frames used internally by multi-step
   tracking were silently left unrotated (unnoticed because `sample.sh`
   uses `traj_int=1`). Windowing makes rotating everything the natural
   choice.
2. The rotation fill value is NaN, not a float32 sentinel. v1's BICUBIC
   interpolation mixed a huge sentinel (~1e38) with real data at the
   rotation's boundary ring, producing values that didn't equal `zmiss`
   exactly and so were silently treated as valid. NaN propagates through
   BICUBIC and is always treated as missing by pyvttrac.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def rotate_window(z_base: np.ndarray, t_win: np.ndarray, i0: int, omega: float, out: "np.ndarray | None" = None) -> np.ndarray:
    """Rotate each frame in `z_base` (nwin, ny, nx) by `omega` (rad/s) times
    its elapsed time from frame `i0`. Frame `i0` itself is left untouched
    (elapsed time 0 -> a 0-degree rotation is a no-op, but we skip the PIL
    round-trip for it entirely).

    PIL's positive angle is counter-clockwise, about the image center.
    """
    nwin, ny, nx = z_base.shape
    if out is None:
        out = np.empty_like(z_base)
    deg_per_sec = np.rad2deg(omega)
    for k in range(nwin):
        if k == i0:
            out[k] = z_base[k]
            continue
        dt = t_win[k] - t_win[i0]
        img = Image.fromarray(z_base[k], mode="F")
        out[k] = np.asarray(img.rotate(deg_per_sec * dt, resample=Image.BICUBIC, fillcolor=np.nan))
    return out
