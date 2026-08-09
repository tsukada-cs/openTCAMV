"""Concatenating `10_conduct_tracking.py` outputs along `time`.

Not a speed tool: `10_` already tracks every Omega and ns.
within a single process/file. This exists for cases where tracking itself
was split across processes/machines by time range (memory limits, cluster
distribution) and needs stitching back together before `20_
finalize_tracking.py`. For within-process speed, use `10_`'s `--workers`
(OpenMP thread count) instead.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def resolve_filenames(target: "list[str] | None", glob_strings: "str | None", exclude_texts: "list[str] | None") -> pd.Series:
    if target is not None:
        fnames = pd.Series(target)
    elif glob_strings is not None:
        fnames = pd.Series(sorted(glob.glob(glob_strings)))
    else:
        raise ValueError("Please specify either --target or --glob_strings")

    if len(fnames) == 0:
        raise FileNotFoundError(f"No files found with the pattern: {glob_strings}")

    if exclude_texts:
        for except_text in exclude_texts:
            fnames = fnames[~fnames.str.contains(except_text)]

    if len(fnames) >= 20:
        logger.warning("Too many files to concat. Please make sure that is what you want.")

    return fnames


def concat_along_time(fnames: "list[str]", drop_vars: "list[str] | None" = None) -> xr.Dataset:
    """Concatenates `10_` output files along `time` (works unchanged whether
    or not each file carries an `omega`/`ns` dimension -- those are never
    the concat axis here)."""
    all_ds = None
    for fname in fnames:
        ds = xr.open_dataset(fname).swap_dims({"it": "time"})
        if drop_vars:
            ds = ds.drop_vars(drop_vars)
        all_ds = ds if all_ds is None else xr.concat([all_ds, ds], "time")

    all_ds = all_ds.drop_duplicates("time")
    ditstep = all_ds["it"][1] - all_ds["it"][0]
    all_ds["it"] = xr.DataArray(
        np.arange(all_ds["it"].min(), all_ds["it"].min() + (all_ds["it"].size - 1) * ditstep + ditstep, ditstep),
        dims="time",
    )
    return all_ds.swap_dims({"time": "it"})


def default_oname(glob_strings: "str | None") -> str:
    if glob_strings is None:
        raise ValueError("-o/--oname is required when files are given via --target rather than --glob_strings")
    return f"{Path(glob_strings).resolve().parent}/{Path(glob_strings).name.replace('*', '_concat')}"
