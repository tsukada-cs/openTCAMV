#%%
"""
# openTCAMV -- 20_finalize_tracking.py
This script finalizes the tracking results by selecting the best solution from the candidate solutions.
The candidate solutions are generated by the script `10_conduct_tracking.py`.

## Data requirement
Input file: NetCDF files generated by `10_conduct_tracking.py`

## Usage
# Example 1
$ python 20_finalize_tracking.py ../sample/2017_Lan_ns7_nt1_rot<omega>.nc --ns 7 --omega 0.0000 0.0005 0.0010 0.0015 0.0020 0.0025 --cthmax 10 --exclude stf stb score_ary psr -o ../sample/2017_Lan_ns7_nt1_ref.nc

# Example 2 (with reference flows to apply striations treatment using the result of Example 1)
$ python 20_finalize_tracking.py ../sample/2017_Lan_ns7_nt1_rot<omega>.nc --ns 7 --omega 0.0000 0.0005 0.0010 0.0015 0.0020 0.0025 --cthmax 10 --exclude stf stb score_ary psr --ref_flows ../sample/2017_Lan_ns7_nt1_ref.nc --omega_stri 0.0015 -o ../sample/2017_Lan_ns7_nt1.nc

## Reference
Tsukada, T., Horinouchi, T., & Tsujino, S. (2024). Wind distribution in the eye of tropical cyclone revealed by a novel atmospheric motion vector derivation. Journal of Geophysical Research: Atmospheres, 129, e2023JD040585. https://doi.org/10.1029/2023JD040585
"""


import os
import logging
import argparse

import numpy as np
import xarray as xr
import scipy.ndimage as ndi
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s %(levelname)s %(name)s %(lineno)d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger.info(f"PID: {os.getpid()}")


parser = argparse.ArgumentParser(description="Finalize tracking results")
parser.add_argument("ifns_rule", type=str, help="filename format used in f-string using <ns> as template size and <omega> as Ω_i")
parser.add_argument("--ns", type=int, nargs="+", help="ns list")
parser.add_argument("--omega", type=str, nargs="+", help="Ω_i list")
parser.add_argument("-o", "--ofn", type=str, help="output filename")
parser.add_argument("--Tw", type=float, default=1210, help="time period of median filtering window (in sec)")
parser.add_argument("--Hw", type=float, default=6, help="width & height of median filtering window for Cartesian grid (in km)")
parser.add_argument("--xw", type=float, help="x width of median filtering window for Cartesian grid (in km)")
parser.add_argument("--yw", type=float, help="y height of median filtering window for Cartesian grid (in km)")
parser.add_argument("--rw", type=float, default=6, help="r width of median filtering window for polar grid (in km)")
parser.add_argument("--aw", type=float, default=30, help="azimuth width of median filtering window for polar grid (in degree)")
parser.add_argument("--dth", type=float, default=10, help="threshold to limit the maximum d2 for rejection of candidates (in m/s)")
parser.add_argument("--dc", type=float, default=0.5, help="coefficient to be multiplied by the L2 norm of median wind speed for rejection of candidates (0: all rejected, 0.5: half length, 1: same length)")
parser.add_argument("--useV", action="store_true", help="if specified, use absolute wind speed for median filtering, not using each velocity component")
parser.add_argument("-e", "--max_epoch", type=float, default=20, help="the maximum number of epochs for updating the solutions")
parser.add_argument("--IRdiff", type=str, metavar="B13-B14", help="the formula for IR difference. It evaluates the difference between two variables parsed with a minus sign. The grid with |IR1-IR2| >= --dIR are masked")
parser.add_argument("--dIR", type=float, default=2, help="threshold to limit maximum IR difference (in K). The values larger than it will be masked.")
parser.add_argument("--cth", type=str, default="cth", help="cloud top height variable name")
parser.add_argument("--cthmin", type=float, help="threshold to limit the minimum cloud top height at initial position (in km)")
parser.add_argument("--cthmax", type=float, help="threshold to limit the maximum cloud top height at initial position (in km)")
parser.add_argument("--score_th", type=float, help="threshold to limit the minimum score of the candidates")
parser.add_argument("--out_final_omega", action="store_true", help="output `final_omega` which is the used omega values in final solutions")
parser.add_argument("--out_final_ns", action="store_true", help="output `final_ns` which is the used ns in final solutions")
parser.add_argument("--out_final_medians", action="store_true", help="output `final_medians` which is the final median wind speeds")
parser.add_argument("--priority", type=str, default="score", choices=["score","dangv"], help="priority for selecting the best solution. `score` selects the solution with the highest score. `dangv` selects the solution with the smallest angular velocity difference from the reference angular velocity")
parser.add_argument("--exclude", type=str, nargs="*", help="drop specified variables from the final output file")
parser.add_argument("--ref_flows", type=str, help="Reference flows to apply omega_stri procedure")
parser.add_argument("--tau_stri", type=int, default=24, help="temporal window size (in index space) to compute the velocity contrast of --ref_flows for detection of striations")
parser.add_argument("--v_stri", type=float, default=20, help="threshold to limit the maximum velocity contrast for detection of striations (m/s)")
parser.add_argument("--cth_stri", type=int, help="threshold to limit the maximum cloud top height to get reference contrast")
parser.add_argument("--omega_stri", type=float, default=0.0, help="threshold to limit the minimum Ω_i for the striation grids (rad/s)")

# sample
sample_dir = f"{os.path.dirname(__file__)}/../sample"
test_args = f"{sample_dir}/2017_Lan_ns7_nt1_rot<omega>.nc --ns=7 --omega 0.0000 0.0005 0.0010 0.0015 0.0020 0.0025 --cthmax=10 --exclude stf stb score_ary psr -o={sample_dir}/2017_Lan_ns7_nt1_ref.nc".split()
# test_args = f"{sample_dir}/2017_Lan_ns7_nt1_rot<omega>.nc --ns=7 --omega 0.0000 0.0005 0.0010 0.0015 0.0020 0.0025 --cthmax=10 --exclude stf stb score_ary psr --ref_flows={sample_dir}/2017_Lan_ns7_nt1_ref.nc --omega_stri=0.0015 -o {sample_dir}/2017_Lan_ns7_nt1.nc".split()

try:
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
    args = parser.parse_args(test_args)
except:
    args = parser.parse_args()

logger.info(f"[Input filename rule] {args.ifns_rule}")
logger.info(f"[List of ns] {args.ns}")
logger.info(f"[List of Ω_i] {args.omega}")
logger.info(f"[Scheduled output filename] {args.ofn}")
#%% Load data and preprocess
fn_list = [args.ifns_rule.replace("<ns>",str(ns)).replace("<omega>",omega) for ns in args.ns for omega in args.omega]
flows0 = xr.open_dataset(fn_list[0])
if not flows0.attrs["polar"]:
    dim1, dim2, loc1, loc2, v1, v2 = "y", "x", "yloc", "xloc", "vy", "vx"
else:
    dim1, dim2, loc1, loc2, v1, v2 = "r", "a", "rloc", "aloc", "vr", "vt"

keepvars = [key for key in flows0.data_vars.keys()]
used_vars = [v1, v2, loc1, loc2, "score"]
for var in used_vars+args.exclude:
    if var in keepvars:
        keepvars.remove(var)

flows_org = xr.open_mfdataset(fn_list, concat_dim=["omega"], combine="nested")[used_vars]
flows_org["omega"] = np.array(args.omega).astype(float)
if "ns" not in flows_org.dims:
    flows_org[used_vars] = flows_org[used_vars].expand_dims("ns", axis=0)
flows_org["ns"] = np.array(args.ns)

for var in keepvars:
    flows_org[var] = flows0[var]

if args.IRdiff:
    IR1, IR2 = args.IRdiff.split("-")

# Median filter window size
args.xw = args.xw or args.Hw
args.yw = args.yw or args.Hw
iTw = int(args.Tw//flows0.attrs["dtmean"])
ixw = int(args.xw//flows0.attrs["xint"])
iyw = int(args.yw//flows0.attrs["yint"])
windows_sizes = {"it":(iTw//2)*2+1, dim1:(ixw//2)*2+1, dim2:(iyw//2)*2+1}
logger.info(f"[Median filter window] {windows_sizes}")
#%%
def squeeze_cth(flows, minmax="max"):
    if minmax == "max":
        if f"{args.cth}max" in flows:
            cth = flows[f"{args.cth}max"]
        elif "it_rel" in flows[args.cth].dims:
            cth = flows[args.cth].max("it_rel")
        else:
            cth = flows[args.cth]
    elif minmax == "min":
        if f"{args.cth}min" in flows:
            cth = flows[f"{args.cth}min"]
        elif "it_rel" in flows[args.cth].dims:
            cth = flows[args.cth].min("it_rel")
        else:
            cth = flows[args.cth]
    return cth

valid_index = xr.ones_like(flows0.vx, dtype=bool)
if args.cthmin:
    valid_index *= squeeze_cth(flows0, "min") >= args.cthmin
if args.cthmax:
    valid_index *= squeeze_cth(flows0, "max") <= args.cthmax
if args.IRdiff:
    valid_index *= flows0[IR1]-flows0[IR2] <= args.dIR
flows_org[used_vars] = flows_org[used_vars].where(valid_index)

if args.score_th:
    flows_org[used_vars] = flows_org[used_vars].where(flows_org["score"].max("it_rel_v") <= args.score_th)

if args.priority == "score":
    flows_org["mscore"] = -flows_org["score"].mean("it_rel_v")
    args.priority = "mscore"
elif args.priority == "dangv":
    if not flows_org.attrs["polar"]:
        r2d = np.hypot(flows_org.x, flows_org.y)
        theta2d = np.arctan2(flows_org.y, flows_org.x)
        angv = (-flows_org["vx"]*np.sin(theta2d) + flows_org["vy"]*np.cos(theta2d))/r2d/1000
    else:
        angv = flows_org["vt"]/flows_org["r"]/1000
    flows_org["dangv"] = np.abs(angv-flows_org["omega"])
    flows_org["omega"] = np.arange(flows_org["omega"].size)
    del angv

if args.omega_stri and args.ref_flows:
    used_vars_in_ref = [v1, v2]
    if args.cth in flows_org:
        used_vars_in_ref.append(args.cth)
    ref_flows = xr.open_dataset(args.ref_flows)[used_vars_in_ref]
    ref_flows["v"] = np.hypot(ref_flows[v1],ref_flows[v2])

    rolling_obj = ref_flows["v"].rolling(it=args.tau_stri, center=True, min_periods=1)
    v_contrast = rolling_obj.max("it") - rolling_obj.min("it")

    striation_grid = (v_contrast >= args.v_stri)
    if args.cth_stri:
        cthmax = squeeze_cth(ref_flows, "max")
        striation_grid *= (cthmax <= args.cth_stri)
    invalid_index_considering_stri = (np.abs(flows_org.omega) < np.abs(args.omega_stri)) * striation_grid
    flows_org[used_vars] = flows_org[used_vars].where(~invalid_index_considering_stri)
#%% sort by score along omega
if args.useV:
    flows_org["v"] = np.hypot(flows_org[v1],flows_org[v2])
    target_flows = flows_org[["v",args.priority]].copy(deep=True).compute()
else:
    if flows_org.attrs["polar"]:
        flows_org["vx"] = -flows_org["vt"]*np.sin(flows_org.a) + flows_org["vr"]*np.cos(flows_org.a)
        flows_org["vy"] = flows_org["vt"]*np.cos(flows_org.a) + flows_org["vr"]*np.sin(flows_org.a)
    target_flows = flows_org[["vx","vy",args.priority]].copy(deep=True).compute()

omega_axis = 1
index_sortby_priority = np.argsort(target_flows[args.priority], axis=omega_axis)
for varname in target_flows:
    target_flows[varname].data = np.take_along_axis(target_flows[varname].data, index_sortby_priority.values, axis=omega_axis)
target_flows = target_flows.rename({"omega": "priority"})
# %% Limit the score to the minimum valid `ns`
ns_broadcast = target_flows["ns"].broadcast_like(target_flows[args.priority])
ns_broadcast = ns_broadcast.where(target_flows[args.priority].notnull(), np.nan)
valid_ns_mins = ns_broadcast.min("ns").compute()
del ns_broadcast

ns_min = flows_org.ns.min().item()
target_flows = target_flows.sel(ns=valid_ns_mins.fillna(ns_min)).compute()
priority_axis = 0
#%% Update the solution by median filtering
shape = target_flows[args.priority].shape
final_indices = np.arange(flows_org.omega.size)[:,None,None,None].repeat(shape[1],1).repeat(shape[2],2).repeat(shape[3],3)
for ep in range(args.max_epoch):
    logger.info(f"===> Updating {ep+1} / {args.max_epoch}")
    if args.useV:
        v_med = target_flows["v"].isel(priority=0).rolling(windows_sizes, min_periods=1, center=True).median()
        d2 = np.abs(target_flows["v"]-v_med)
    else:
        vx_med = target_flows["vx"].isel(priority=0).rolling(windows_sizes, min_periods=1, center=True).median()
        vy_med = target_flows["vy"].isel(priority=0).rolling(windows_sizes, min_periods=1, center=True).median()
        d2 = np.hypot(target_flows["vx"]-vx_med, target_flows["vy"]-vy_med).compute()

    valid_index = xr.ones_like(d2, bool)
    if args.dth:
        valid_index *= (d2 <= args.dth)
    if args.dc:
        if args.useV:
            valid_index *= (d2 <= v_med*args.dc)
        else:
            valid_index *= (d2 <= np.hypot(vx_med,vy_med)*args.dc)
    target_flows[args.priority] = target_flows[args.priority].where(valid_index)
    
    index_sortby_priority = np.argsort(target_flows[args.priority], axis=priority_axis)
    if (index_sortby_priority == final_indices).all().item():
        break
    for varname in target_flows:
        target_flows[varname].data = np.take_along_axis(target_flows[varname].data, index_sortby_priority.values, axis=priority_axis)

if args.out_final_medians:
    if args.useV:
        target_flows["v_med"] = v_med
    else:
        target_flows["vx_med"] = vx_med
        target_flows["vy_med"] = vy_med

final_score = target_flows[args.priority].isel(priority=0).compute().drop_vars("priority")
valid_omega = target_flows[args.priority].notnull().any("priority").compute()
del target_flows

#%% finalize using `final_score` & `valid_omega`
logger.info(f"Finalizing...")
if flows_org.ns.size == 1:
    flows_ns_min = flows_org.squeeze()
else:
    flows_ns_min = flows_org.sel(ns=valid_ns_mins.fillna(ns_min))
if args.out_final_ns:
    flows_ns_min["final_ns"] = valid_ns_mins

corresponding_omega_is_zero = np.abs(flows_ns_min[args.priority] - final_score)
final_omega = corresponding_omega_is_zero.fillna(np.inf).idxmin("omega")
final_flows = flows_ns_min[used_vars].sel(omega=final_omega).where(valid_omega).compute()

for keepvar in keepvars:
    final_flows[keepvar] = flows_ns_min[keepvar]
if args.out_final_omega:
    final_flows["final_omega"] = final_omega

logger.info(f"[FINALIZED] at epoch={ep+1}")
#%% output
for varname in final_flows:
    if "it_rel" in final_flows[varname].dims:
        final_flows[varname] = final_flows[varname].transpose("it", "it_rel", ...)
    elif varname in ("score","psr"):
        final_flows[varname] = final_flows[varname].transpose("it", "it_rel_v", ...)
    elif varname == "score_ary":
        final_flows[varname] = final_flows[varname].transpose("it", "it_rel_v", "scy", "scx", ...)
    if "missing_value" in final_flows[varname].encoding:
        final_flows[varname].encoding["_FillValue"] = final_flows[varname].encoding["missing_value"]
final_flows.to_netcdf(args.ofn)
logger.info(f"[SUCCESS] {args.ofn}")
#%%
