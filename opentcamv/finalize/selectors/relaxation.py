"""Relaxation-labeling candidate selector: interface skeleton only. 
Not implemented -- `--selector relaxation` exists on the CLI so the 
option is discoverable and its future home is clear, but selecting it 
raises `NotImplementedError`.

Classical relaxation labeling (Rosenfeld-Hummel-Zucker), mapped onto the
current hard-assignment selector (`iterative_median`) it would eventually
sit alongside:

| Relaxation labeling          | `iterative_median` (hard assignment)              |
|-------------------------------|----------------------------------------------------|
| Label set                     | Candidate set (omega x ns)                          |
| Initial probability `p_i(l)`  | Rank from cost (rank 0 -> probability 1, rest 0)    |
| Neighborhood `N(i)`            | `rolling(windows_sizes, center=True)` window        |
| Compatibility `r_ij(l, l')`   | Closeness of `d2` to the neighborhood median/mean   |
| Update rule                   | Reject candidates with `d2 > dth` or `> |v_med|*dc` |
| Convergence                   | Sort order unchanged epoch-to-epoch                 |

A real implementation would keep `SelectionResult.weight` (per-candidate
probability) alive across iterations instead of collapsing to a single
winner up front, using `.base.neighborhood_reduce`/`.base.candidate_distance`/
`.base.iterate_until_stable` as its building blocks.
"""

from __future__ import annotations

from .base import SelectionResult


class RelaxationSelector:
    """Skeleton `Selector` implementation. Every method raises
    `NotImplementedError`; the class exists so `--selector relaxation` has
    a real (if inert) target and the `Selector` Protocol has a second
    implementer to check itself against."""

    def select(self, cand, *, window: dict, max_epoch: int) -> SelectionResult:
        raise NotImplementedError(
            "The relaxation-labeling selector is not implemented yet. "
            "Use --selector iterative_median."
        )
