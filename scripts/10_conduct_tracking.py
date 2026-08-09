#%%
"""
# openTCAMV -- 10_conduct_tracking.py
This script performs the cloud tracking for satellite imagery.
The tracking is conducted using the `pyVTTrac` library, which is a cloud tracking algorithm based on the template matching method.
Core logic lives in the `opentcamv` package; this script is a thin CLI driver.

## Data requirement
Input file: NetCDF file that contains a tracked variable with (time, y, x) dimensions.
The x and y dimensions should have "units" attribute with "km".

## Example usage
$ python 10_conduct_tracking.py ../sample/2017_Lan_aeqd_sample.nc --revrot 0.0000 0.0005 0.0010 0.0015 0.0020 0.0025 --varname=B03 --ns=7 --ntrac=1 --Sth0=0.7 -o='../sample/2017_Lan_ns7_nt1.nc' --ygran=-45:45 --xgran=-45:45 --traj_int=1 --Vs=10 --record_initpos cth B03 B13 B14 --out_cthmax --Vc=20 --Vd=20 --Td=60 --Vth=5

With multiple --revrot values, the default output is a single file holding
all Omegas along an `omega` dimension. Pass --split_omega for one file per
Omega instead (then -o must contain a `<omega>` placeholder).

## Reference
Tsukada, T., Horinouchi, T., & Tsujino, S. (2024). Wind distribution in the eye of tropical cyclone revealed by a novel atmospheric motion vector derivation. Journal of Geophysical Research: Atmospheres, 129, e2023JD040585. https://doi.org/10.1029/2023JD040585
"""

# Main part
import os
import logging

import numpy as np

import opentcamv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s %(name)s %(lineno)d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger.info(f"PID: {os.getpid()}")

parser = opentcamv.cli.build_parser()

sample_dir = f"{os.path.dirname(__file__)}/../sample"
test_args = f"{sample_dir}/2017_Lan_aeqd_sample.nc --revrot 0.0000 0.0005 0.0010 0.0015 0.0020 0.0025 \
    --ns=7 --ntrac=1 --Sth0=0.7 -o={sample_dir}/2017_Lan_ns7_nt1.nc --varname=B03 \
    --ygran=-45:45 --xgran=-45:45 --traj_int=1 --Vs=10 \
    --record_initpos cth B03 B13 B14 --out_cthmax \
    --Vc=20 --Vd=20 --Td=60 --Vth=5".split()

try:
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
    args = parser.parse_args(test_args)
except NameError:
    args = parser.parse_args()
args = opentcamv.cli.normalize_args(args)

logger.info(f"[{os.getpid()}] {args}")

#%% Input file open and select
frames, tname, yname, xname = opentcamv.data.open_frames(args)
mask = opentcamv.data.build_mask(args, frames)
nt = frames[tname].size
tg = opentcamv.data.compute_tg(args, nt)

setup = opentcamv.params.TrackingSetup.from_args(args, frames, args.varname)
names = opentcamv.output.axis_names(args)
xxg, yyg = opentcamv.output.build_seed_positions(args)
polar_trig = opentcamv.output.polar_trig(args)
costh, sinth = polar_trig if polar_trig is not None else (None, None)

coords = opentcamv.output.build_coords(args, frames, tname, tg)
pickup_it_rel, pickup_it_rel_v = opentcamv.output.pickup_indices(args, coords)

#%% Build one empty output Dataset per Omega
ofls, encodings = {}, {}
for omega in args.revrot:
    ofl, encoding = opentcamv.output.build_empty_dataset(args, frames, tname, coords, tg, setup)
    ofls[omega] = ofl
    encodings[omega] = encoding

# `--dtlimit` gate, evaluated once per tid0 (shared across Omega, since it
# only depends on the it_rel-spaced output time grid, not on rotation).
z = frames[args.varname].astype(np.float32).values
t = opentcamv.data.build_time_seconds(frames, tname)
grid = setup.grid
tracker = setup.make_tracker()
diagnostics = opentcamv.tracking.diagnostics_arg(args)

#%% Perform tracking: tid0 on the outside, Omega on the inside (shared window slicing/cache locality)
for j, tid0 in enumerate(tg.tolist()):
    if (j % 10) == 0 or j == tg.size - 1:
        logger.info(f"[{os.getpid()}] Processing: {j + 1}/{tg.size}")

    lo, hi = opentcamv.tracking.window_bounds(tid0, setup, nt)
    t_win = t[lo:hi + 1]
    if hi > lo and np.max(np.abs(np.diff(t_win))) >= args.dtlimit:
        logger.info(f"[{os.getpid()}] Max time interval is exceeded at {j}th iteration")
        continue

    z_base = z[lo:hi + 1]
    mask_win = None if mask is None else mask[lo:hi + 1]
    t0_local = tid0 - lo

    for omega in args.revrot:
        z_win = opentcamv.rotation.rotate_window(z_base, t_win, t0_local, omega) if omega else z_base
        fwd, bwd = opentcamv.tracking.track_at(
            tracker, setup, z_win, t_win, mask_win, grid, t0_local, lo, xxg, yyg, diagnostics=diagnostics
        )
        step = opentcamv.aggregate.aggregate_step(args, setup, fwd, bwd, tid0, t, grid, pickup_it_rel, pickup_it_rel_v)

        ofl = ofls[omega]
        if not args.polar:
            ofl["vx"].data[j] = step.vx
            ofl["vy"].data[j] = step.vy
            ofl["xloc"].data[j] = step.xtraj
            ofl["yloc"].data[j] = step.ytraj
        else:
            vr, vt = opentcamv.aggregate.to_polar(step.vx, step.vy, costh, sinth)
            rtraj, atraj = opentcamv.aggregate.traj_to_polar(step.xtraj, step.ytraj)
            ofl["vr"].data[j] = vr
            ofl["vt"].data[j] = vt
            ofl["rloc"].data[j] = rtraj
            ofl["aloc"].data[j] = atraj

        ofl["score"].data[j] = step.score
        if setup.forward:
            ofl["stf"].data[j] = step.stf
        if setup.backward:
            ofl["stb"].data[j] = step.stb
        if args.ward == "bothward":
            ofl["vxfm"].data[j] = step.vxfm
            ofl["vyfm"].data[j] = step.vyfm
            ofl["vxbm"].data[j] = step.vxbm
            ofl["vybm"].data[j] = step.vybm

        if args.out_subimage:
            ofl["zss"].data[j] = step.zss
        if args.out_score_ary:
            ofl["score_ary"].data[j] = step.score_grids

#%% Per-Omega postprocessing (revrot restore -> screening -> records -> attrs)
processed = {}
for omega, ofl in ofls.items():
    opentcamv.screening.restore_revrot(ofl, args, omega, names)
    ofl = opentcamv.screening.screen(ofl, args, names)

    ofl = opentcamv.records.record_initpos(ofl, args, frames, tname, yname, xname, tg)
    ofl = opentcamv.records.record_alongtraj_and_cth(ofl, args, frames, tname, yname, xname, xxg, yyg)

    if args.out_score_ary and args.out_psr:
        around_ratio = 0.15
        ofl["psr"] = opentcamv.records.peak_to_sidelobe_ratio(ofl, around_ratio)
        ofl["psr"].attrs.update({"long_name": "peak-to-sidelobe ratio", "around_ratio": around_ratio, "units": ""})

    ofl.attrs.update(opentcamv.output.build_attrs(args, omega, script_path=__file__))
    ofl.attrs.update(setup.to_attrs(z, omega, mask))
    processed[omega] = ofl

#%% Write: one file per Omega (--split_omega, v1-compatible) or a single
# combined file with an `omega` dimension (default; a no-op distinction
# when there's only one Omega, which never gets an `omega` dimension).
if args.split_omega or len(processed) == 1:
    for omega, ofl in processed.items():
        ofn = args.ofn.replace("<omega>", f"{omega:.4f}") if "<omega>" in args.ofn else args.ofn
        opentcamv.output.write(ofl, encodings[omega], ofn, args.complevel)
        logger.info(f"[{os.getpid()}] SUCCESS: {ofn}")
else:
    combined = opentcamv.output.combine_omega(processed, record_initpos=args.record_initpos)
    opentcamv.output.write(combined, {}, args.ofn, args.complevel)
    logger.info(f"[{os.getpid()}] SUCCESS: {args.ofn}")
# %%
