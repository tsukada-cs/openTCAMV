"""Synthetic candidate-file generator shared by the `finalize/` (20_
finalize_tracking.py) test suite.

Each omega gets its own file, matching what `10_conduct_tracking.py`
actually produces (the schema `20_finalize_tracking.py`/`finalize/` reads):
`vx`/`vy`/`xloc`/`yloc`/`score` (`used_vars`, or `vr`/`vt`/`rloc`/`aloc` in
`--polar` mode), plus `cth`/`cthmax`/`B13`/`B14` (`--record_initpos`/
`--out_cthmax` outputs, part of `keepvars`), with `polar`/`dtmean`/`xint`/
`yint`/`rint`/`nath` attrs.

The synthetic field has a deterministic, spatially-varying "true" candidate:
at grid point (i, j), `true_omega_idx = (i + j) % n_omega` is the
bias-free, high-score candidate; every other omega at that point gets a
growing bias and a lower score. This makes "the median-filtering selector
must recover a spatially-varying answer" an actual property of the test
data, not just a schema fixture. A couple of "trap" points additionally
give an outlier candidate a higher score than the correct one, so a single
sort-by-score picks the wrong candidate and only iterative rejection
recovers the right answer -- exercising that logic, not just the trivial
"highest score already correct" case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def make_finalize_candidates(
    tmp_path,
    *,
    n_omega: int = 4,
    ny: int = 9,
    nx: int = 9,
    nit: int = 6,
    seed: int = 0,
    bias_scale: float = 8.0,
    cthmax_base: float = 5.0,
    ns: int = 7,
    polar: bool = False,
    fname_prefix: str = "cand",
) -> "tuple[str, list[str], list[float], list[str]]":
    """Write one NetCDF file per omega under `tmp_path`. Returns
    `(ifns_rule, omega_strs, omegas, paths)` -- `ifns_rule` uses `<ns>`/
    `<omega>` placeholders the same way `10_conduct_tracking.py`'s `-o`
    does, so it's directly usable as a `20_finalize_tracking.py` CLI arg."""
    rng = np.random.RandomState(seed)
    omegas = [round(0.0005 * i, 4) for i in range(n_omega)]
    omega_strs = [f"{om:.4f}" for om in omegas]

    if not polar:
        dim1_name, dim2_name, loc1_name, loc2_name, v1_name, v2_name = "y", "x", "yloc", "xloc", "vy", "vx"
        d1 = np.arange(ny, dtype=np.float64) - ny // 2
        d2 = np.arange(nx, dtype=np.float64) - nx // 2
        d1_units, d2_units = "km", "km"
    else:
        dim1_name, dim2_name, loc1_name, loc2_name, v1_name, v2_name = "r", "a", "rloc", "aloc", "vr", "vt"
        d1 = 4.0 + np.arange(ny, dtype=np.float64)  # radius, km
        d2 = np.linspace(0, 2 * np.pi, nx, endpoint=False)  # azimuth, radian
        d1_units, d2_units = "km", "radian"

    n1, n2 = d1.size, d2.size
    v1_true = 5.0 + 0.3 * d1[:, None] * np.ones((1, n2))
    v2_true = -2.0 + 0.2 * d2[None, :] * np.ones((n1, 1))
    true_omega_idx = (np.arange(n1)[:, None] + np.arange(n2)[None, :]) % n_omega

    trap_mask = np.zeros((n1, n2), dtype=bool)
    trap_omega_idx = np.full((n1, n2), -1, dtype=int)
    if n_omega >= 2 and n1 >= 4 and n2 >= 4:
        for (t1, t2) in [(2, 2), (n1 - 3, n2 - 3)]:
            trap_mask[t1, t2] = True
            trap_omega_idx[t1, t2] = (true_omega_idx[t1, t2] + 1) % n_omega

    it = np.arange(nit)
    time = pd.Timestamp("2020-01-01") + pd.to_timedelta(it * 150, unit="s")
    it_rel = np.array([-1, 0, 1])
    it_rel_v = np.array([-0.5, 0.5])

    ifn_paths = []
    for om_idx, om in enumerate(omegas):
        is_true = (true_omega_idx == om_idx)
        bias = np.where(is_true, 0.0, bias_scale * (1 + np.abs(om_idx - true_omega_idx)))
        noise1 = rng.normal(0, 0.05, size=(nit, n1, n2))
        noise2 = rng.normal(0, 0.05, size=(nit, n1, n2))

        v1 = v1_true[None] + bias[None] + noise1
        v2 = v2_true[None] + bias[None] * 0.5 + noise2
        score = np.where(is_true[None], 0.92, 0.55 - 0.03 * np.abs(om_idx - true_omega_idx)[None]) \
            + rng.normal(0, 0.01, size=(nit, n1, n2))
        score = np.broadcast_to(score, (nit, n1, n2)).copy()

        is_trap_winner = trap_mask & (trap_omega_idx == om_idx)
        score[:, is_trap_winner] = 0.97  # outranks the true candidate's 0.92

        cthmax = cthmax_base + 1.5 * om_idx + rng.normal(0, 0.2, size=(nit, n1, n2))
        cth = cthmax - 1.0
        b13 = 250.0 + rng.normal(0, 1.0, size=(nit, n1, n2))
        b14 = b13 - 1.0 + rng.normal(0, 0.5, size=(nit, n1, n2))

        loc1 = np.broadcast_to(d1[None, None, :, None], (nit, len(it_rel), n1, n2)).astype(np.float64).copy()
        loc2 = np.broadcast_to(d2[None, None, None, :], (nit, len(it_rel), n1, n2)).astype(np.float64).copy()
        score_v = np.broadcast_to(score[:, None], (nit, len(it_rel_v), n1, n2)).astype(np.float32).copy()

        it_plus_it_rel = it[:, None] + it_rel[None, :]
        it_plus_it_rel = np.clip(it_plus_it_rel, 0, nit - 1)
        time2 = time.values[it_plus_it_rel]

        ds = xr.Dataset(
            data_vars={
                v1_name: (["it", dim1_name, dim2_name], v1.astype(np.float32)),
                v2_name: (["it", dim1_name, dim2_name], v2.astype(np.float32)),
                loc1_name: (["it", "it_rel", dim1_name, dim2_name], loc1.astype(np.float32)),
                loc2_name: (["it", "it_rel", dim1_name, dim2_name], loc2.astype(np.float32)),
                "score": (["it", "it_rel_v", dim1_name, dim2_name], score_v),
                "cth": (["it", dim1_name, dim2_name], cth.astype(np.float32)),
                "cthmax": (["it", dim1_name, dim2_name], cthmax.astype(np.float32)),
                "B13": (["it", dim1_name, dim2_name], b13.astype(np.float32)),
                "B14": (["it", dim1_name, dim2_name], b14.astype(np.float32)),
                "time2": (["it", "it_rel"], time2),
            },
            coords={
                "it": ("it", it),
                "time": ("it", time),
                dim1_name: (dim1_name, d1, {"units": d1_units}),
                dim2_name: (dim2_name, d2, {"units": d2_units}),
                "it_rel": ("it_rel", it_rel),
                "it_rel_v": ("it_rel_v", it_rel_v),
            },
            attrs=dict(
                polar=int(polar), dtmean=150.0, xint=1.0, yint=1.0, rint=1.0, nath=nx,
                itstep=1, ntrac=1, nsx=ns, nsy=ns,
            ),
        )
        # Filename matches 20_'s own `ifns_rule.replace("<ns>", str(ns)).replace("<omega>", omega_str)`
        # convention exactly, so `ifns_rule` above is usable as a real `20_` CLI argument.
        p = tmp_path / f"{fname_prefix}_ns{ns}_rot{omega_strs[om_idx]}.nc"
        ds.to_netcdf(p)
        ifn_paths.append(str(p))

    ifns_rule = str(tmp_path / f"{fname_prefix}_ns<ns>_rot<omega>.nc")
    return ifns_rule, omega_strs, omegas, ifn_paths
