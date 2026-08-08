"""Argument parser for `10_conduct_tracking.py`, plus post-parse normalization.

The parser is a straight port of v1's, plus: `--revrot` becomes `nargs="+"`
(looped over in a single process instead of one process per Omega), and
`--workers` / `--subgrid` / `--method` are new.
"""

from __future__ import annotations

import argparse


def _parse_slice(s):
    a = [int(e) if e.strip() else None for e in s.split(":")]
    return slice(*a)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Conduct cloud tracking for satellite imagery")
    parser.add_argument("ifn", type=str, help="file path to input NetCDF file")
    parser.add_argument("-s", "--start", type=str, help="start time in yyyymmddTHHMMSS format")
    parser.add_argument("-e", "--end", type=str, help="end time in yyyymmddTHHMMSS format")
    parser.add_argument("-o", "--ofn", default="./tmp.nc", type=str, help="output NetCDF file path. Must contain a `<omega>` placeholder when --revrot has more than one value")
    parser.add_argument("-n", "--ntrac", default=2, type=int, help="The number of tracking for both forward and backward tracking")
    parser.add_argument("--ward", type=str, default="bothward", choices=["bothward", "forward", "backward"], help="time direction for tracking")
    parser.add_argument("--tidstep", default=1, type=int, help="time index interval of initial time for start tracking (1 means every time index)")
    parser.add_argument("--traj_int", default=None, type=int, help="time index interval for output of trajectory")
    parser.add_argument("-v", "--vagg", type=str, default="mean", choices=["org", "mean", "startend"], help="how to aggregate the vectors; 'org' for original vectors without any aggregations, 'mean' for the velocity be averaging vectors, 'startend' for the veclocity by connecting the start and end points")
    parser.add_argument("--polar", action="store_true", help="if specified, use polar coordinates points as initial template positioning, if not use Cartesian grid")
    parser.add_argument("--use_init_temp", action="store_true", help="use initial template through tracking without updating the template")
    parser.add_argument("--no_subgrid", action="store_true", help="alias for --subgrid=none")
    parser.add_argument("--subgrid", type=str, default="paraboloid", choices=["paraboloid", "gaussian", "none"], help="subgrid peak-refinement method")
    parser.add_argument("--method", type=str, default="xcor", choices=["xcor", "ncov"], help="score method")
    parser.add_argument("--workers", type=int, default=None, help="OpenMP thread count. None = OpenMP default (usually all cores), 1 = sequential")
    parser.add_argument("--itran", type=_parse_slice, help="time-axis colon-separated slice of initial time for start tracking, with higher priority over --start and --end")
    parser.add_argument("--xgran", default=slice(-50, 50), type=_parse_slice, help="x-axis colon-separated slice of initial template positions (with interval of --xint)")
    parser.add_argument("--xint", default=1.0, type=float, help="x-axis interval of initial template positions in the (equally spaced grid)")
    parser.add_argument("--ygran", default=slice(-50, 50), type=_parse_slice, help="y-axis colon-separated slice of initial template positions (with interval of --yint)")
    parser.add_argument("--yint", default=1.0, type=float, help="y-axis interval of initial template positions (equally spaced grid)")
    parser.add_argument("--rgran", default=slice(4, 50), type=_parse_slice, help="r-axis colon-separated slice of initial template positions (equally spaced grid)")
    parser.add_argument("--rint", default=1.0, type=int, help="r-axis interval of initial template positions (equally spaced grid)")
    parser.add_argument("--nath", default=60, type=int, help="number of azimuthal initial template positions in polar coordinates")
    parser.add_argument("--ns", default=11, type=int, help="template size in pixel dimension")
    parser.add_argument("--nsx", type=int, help="template width with higher priority over --ns")
    parser.add_argument("--nsy", type=int, help="template height with higher priority over --ns")
    parser.add_argument("--Vd", default=20.0, type=float, help="threshold to limit the maximum velocity difference between velocities obtained from forward and backward tracking as vectors (available if --ward='bothward' and --vagg='vmean' or 'startend')")
    parser.add_argument("--Td", type=float, help="threshold to limit the maximum angle difference between velocities obtained from forward and backward tracking as vectors (available if --ward='bothward' and --vagg='vmean' or 'startend')")
    parser.add_argument("--Vth", type=float, default=5.0, help="threshold speed for screening with --Td")
    parser.add_argument("--Vs", default=80.0, type=float, help="search range for cloud tracking in velocity dimension (m/s)")
    parser.add_argument("--hs", type=int, help="Search range for cloud tracking in pixel count with higher priority over --Vs")
    parser.add_argument("--Vc", default=20.0, type=float, help="threshold to limit the maximum velocity change between consecutive images")
    parser.add_argument("--vlim", default=0, type=float, help="Threshold to limit the maximum speed (m/s)")
    parser.add_argument("--Sth0", default=0.8, type=float, help="minimum score required for the first-time tracking")
    parser.add_argument("--Sth1", default=0.8, type=float, help="minimum score required for the subsequent tracking")
    parser.add_argument("--Cth", default=3, type=float, help="minimum contrast to track the template")
    parser.add_argument("--peak_inside_th", default=None, type=float, help="")
    parser.add_argument("--itstep", default=1, type=int, help="if >1, skip")
    parser.add_argument("--varname", default="B03", type=str, help="variable name of tracking target")
    parser.add_argument("--maskvar", type=str, help="variable name for creating mask")
    parser.add_argument("--mask_lower_lim", type=float, help="lower limit for mask variable. --maskvar <= --lower_limit will be ignored when scoring")
    parser.add_argument("--mask_upper_lim", type=float, help="upper limit for mask variable. --maskvar >= --upper_limit will be ignored when scoring")
    parser.add_argument("--min_samples", default=1, type=int, help="minimum number of valid values to calculate score when using mask")
    parser.add_argument("--out_subimage", action="store_true", help="if output subimages")
    parser.add_argument("--out_score_ary", action="store_true", help="if output score array")
    parser.add_argument("--out_psr", action="store_true", help="if output Peak-To-Sidelobe ratio of the score field")
    parser.add_argument("--sector", type=str, nargs="*", help="limiting sectors used for tracking")
    parser.add_argument("--dtlimit", default=200.0, type=float, help="specify maximum time interval (in seconds)")
    parser.add_argument("--ref_dt", type=float, help="reference time interval for calculating ixhw and iyhw from Vs (in seconds)")
    parser.add_argument("--revrot", default=[0.0], type=float, nargs="+", help="angular velocity/velocities to rotate images (in rad/s). Positive (negative) value make crockwise (counterclocwise) rotation over time. Multiple values are looped over within a single process (see -o)")
    parser.add_argument("--record_initpos", type=str, nargs="*", help="Record specified variable at their initial position")
    parser.add_argument("--record_alongtraj", type=str, nargs="*", help="Record specified variable along their trajectory")
    parser.add_argument("--cth", type=str, default="cth", help="cloud top height variable name")
    parser.add_argument("--out_cthmin", action="store_true", help="if output minimum cloud top height along each tracking")
    parser.add_argument("--out_cthmax", action="store_true", help="if output maximum cloud top height along each tracking")
    parser.add_argument("--complevel", type=int, default=3, help="compression level for output NetCDF file")
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    """Post-parse normalization shared by all entry points (script + tests)."""
    args.nsx = args.nsx or args.ns
    args.nsy = args.nsy or args.ns
    if args.no_subgrid:
        args.subgrid = "none"
    if len(args.revrot) > 1 and "<omega>" not in args.ofn:
        raise ValueError("`-o/--ofn` must contain an `<omega>` placeholder when --revrot has more than one value")
    if len(args.revrot) > 1 and args.out_score_ary:
        import logging

        logging.getLogger(__name__).warning(
            "Multiple --revrot values with --out_score_ary holds all Omegas' score_ary "
            "buffers in memory at once, which can be very large. Consider running one "
            "Omega per process (as in v1) if you hit memory limits."
        )
    return args
