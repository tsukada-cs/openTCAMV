"""Selector interface (used by the relaxation-labeling selector; the
`iterative_median` implementation predates this and is not yet refactored 
onto it -- see its own module docstring), plus building blocks shared between selectors.

The building blocks below (`neighborhood_reduce`/`candidate_distance`/
`iterate_until_stable`) generalize the three pieces `iterative_median.py`
inlines: the rolling per-neighborhood median/mean, the distance of a
candidate from that neighborhood value, and the epoch loop with a
stability check. `iterative_median.py` is *not* refactored to call these
(see its own docstring for why: its hardcoded axis positions are fragile
enough that touching it at all is deliberately avoided) -- they exist so
`relaxation.py`'s future implementation has the same vocabulary without
duplicating it ad hoc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import numpy as np
import xarray as xr


@dataclass
class SelectionResult:
    label: xr.DataArray  # chosen candidate's identifying coordinate value(s), per (it, dim1, dim2)
    valid: xr.DataArray  # bool, per (it, dim1, dim2)
    cost: xr.DataArray  # chosen candidate's cost, per (it, dim1, dim2)
    weight: "Optional[xr.DataArray]" = None  # per-candidate weight/probability (relaxation labeling only)
    n_epoch: int = 0


class Selector(Protocol):
    def select(self, cand, *, window: dict, max_epoch: int) -> SelectionResult: ...


def neighborhood_reduce(field: xr.DataArray, window: dict, how: str = "median") -> xr.DataArray:
    """Rolling neighborhood aggregate over `window` (a `{dim: size}` map,
    e.g. `windows_sizes` from `finalize.window.window_sizes`), matching how
    `iterative_median.select`'s local-consistency check reduces a candidate
    field (`.rolling(windows_sizes, min_periods=1, center=True)`)."""
    rolling_obj = field.rolling(window, min_periods=1, center=True)
    if how == "median":
        return rolling_obj.median()
    if how == "mean":
        return rolling_obj.mean()
    raise ValueError(f"how must be 'median' or 'mean', got {how!r}")


def candidate_distance(
    cand: "xr.DataArray | tuple[xr.DataArray, ...]", reference: "xr.DataArray | tuple[xr.DataArray, ...]",
) -> xr.DataArray:
    """Distance between a candidate field and a reference field (e.g. its
    `neighborhood_reduce`d value) -- the compatibility measure relaxation
    labeling calls `r_ij(l, l')`. A single `DataArray` pair gives `|cand -
    reference|` (matching `--useV`'s scalar `d2`); same-length tuples of
    components give their Euclidean distance (matching the vector `vx`/`vy`
    `d2 = hypot(...)` case)."""
    if isinstance(cand, tuple):
        return np.sqrt(sum((c - r) ** 2 for c, r in zip(cand, reference)))
    return np.abs(cand - reference)


def iterate_until_stable(step_fn: Callable, state, max_epoch: int):
    """Calls `step_fn(state) -> (state, stable: bool)` up to `max_epoch`
    times, stopping as soon as `stable` is True (matching
    `iterative_median.select`'s "sort order unchanged" convergence check).
    Returns `(state, epoch)`, `epoch` being the 0-indexed iteration the loop
    stopped at."""
    ep = 0
    for ep in range(max_epoch):
        state, stable = step_fn(state)
        if stable:
            break
    return state, ep
