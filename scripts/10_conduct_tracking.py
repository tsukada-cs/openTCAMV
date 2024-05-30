#%%
"""
# openTCAMV -- 10_conduct_tracking.py
This script perform the cloud tracking for satellite imagery.
The tracking is conducted using the `pyVTTrac` library, which is a cloud tracking algorithm based on the template matching method.

## Data requirement
Input file: NetCDF file that contains a tracked variable with (time, y, x) dimensions.
The x and y dimensions should have "units" attribute with "km".

## Example usage
$ python 10_conduct_tracking.py ../sample/2017_Lan_aeqd_sample.nc --revrot 0.0020 --varname=B03 --ns=7 --ntrac=1 --Sth0=0.7 -o=../sample/2017_Lan_ns7_nt1_test.nc --ygran=-45:45 --xgran=-45:45 --traj_int=1 --Vs=10 --record_initpos cth B03 B13 B14 --out_cthmax --Vc=20 --Vd=20 --Td=60 --Vth=5

## Reference
Tsukada, T., Horinouchi, T., & Tsujino, S. (2024). Wind distribution in the eye of tropical cyclone revealed by a novel atmospheric motion vector derivation. Journal of Geophysical Research: Atmospheres, 129, e2023JD040585. https://doi.org/10.1029/2023JD040585
"""

#%% Main part
import os
import logging
import argparse
import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
from PIL import Image

import numpy as np
import pandas as pd
import xarray as xr
import scipy.ndimage as ndi

from pyVTTrac import VTTrac


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s %(levelname)s %(name)s %(lineno)d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger.info(f"PID: {os.getpid()}")


def _parse_slice(s):
    a = [int(e) if e.strip() else None for e in s.split(":")]
    return slice(*a)

# argparse
parser = argparse.ArgumentParser(description="Conduct cloud tracking for satellite imagery")
parser.add_argument("ifn", type=str, help="file path to input NetCDF file")
parser.add_argument("-s", "--start", type=str, help="start time in yyyymmddTHHMMSS format")
parser.add_argument("-e", "--end", type=str, help="end time in yyyymmddTHHMMSS format")
parser.add_argument("-o", "--ofn", default="./tmp.nc", type=str, help="output NetCDF file path")
parser.add_argument("-n", "--ntrac", default=2, type=int, help="The number of tracking for both forward and backward tracking")
parser.add_argument("--ward", type=str, default="bothward", choices=["bothward","forward","backward"], help="time direction for tracking")
parser.add_argument("--tidstep", default=1, type=int, help="time index interval of initial time for start tracking (1 means every time index)")
parser.add_argument("--traj_int", default=None, type=int, help="time index interval for output of trajectory")
parser.add_argument("-v", "--vagg", type=str, default="mean", choices=["org","mean","startend"], help="how to aggregate the vectors; 'org' for original vectors without any aggregations, 'mean' for the velocity be averaging vectors, 'startend' for the veclocity by connecting the start and end points")
parser.add_argument("--polar", action="store_true", help="if specified, use polar coordinates points as initial template positioning, if not use Cartesian grid")
parser.add_argument("--use_init_temp", action="store_true", help="use initial template through tracking without updating the template")
parser.add_argument("--no_subgrid", action="store_true", help="if specified, do not perform a subgrid estimation")
parser.add_argument("--itran", type=_parse_slice, help="time-axis colon-separated slice of initial time for start tracking, with higher priority over --start and --end")
parser.add_argument("--xgran", default=slice(-50,50), type=_parse_slice, help="x-axis colon-separated slice of initial template positions (with interval of --xint)")
parser.add_argument("--xint", default=1.0, type=float, help="x-axis interval of initial template positions in the (equally spaced grid)")
parser.add_argument("--ygran", default=slice(-50,50), type=_parse_slice, help="y-axis colon-separated slice of initial template positions (with interval of --yint)")
parser.add_argument("--yint", default=1.0, type=float, help="y-axis interval of initial template positions (equally spaced grid)")
parser.add_argument("--rgran", default=slice(4,50), type=_parse_slice, help="r-axis colon-separated slice of initial template positions (equally spaced grid)")
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
parser.add_argument("--vlim", default=120.0, type=float, help="Threshold to limit the maximum speed (m/s)")
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
parser.add_argument("--dtlimit", default=200.0, type=float, help="specify maximum dt (in seconds)")
parser.add_argument("--revrot", default=0.0, type=float, help="angular velocity to rotate images (in rad/s). Positive (negative) value make crockwise (counterclocwise) rotation over time")
parser.add_argument("--record_initpos", type=str, nargs="*", help="Record specified variable at their initial position")
parser.add_argument("--record_alongtraj", type=str, nargs="*", help="Record specified variable along their trajectory")
parser.add_argument("--cth", type=str, default="cth", help="cloud top height variable name")
parser.add_argument("--out_cthmin", action="store_true", help="if output minimum cloud top height along each tracking")
parser.add_argument("--out_cthmax", action="store_true", help="if output maximum cloud top height along each tracking")
parser.add_argument("--complevel", type=int, default=3, help="compression level for output NetCDF file")

sample_dir = f"{os.path.dirname(__file__)}/../sample"
test_args = f"{sample_dir}/2017_Lan_aeqd_sample.nc --revrot 0.0020 --ns=7 --ntrac=1 --Sth0=0.7 -o={sample_dir}/2017_Lan_ns7_nt1_test.nc --varname=B03 --ygran=-45:45 --xgran=-45:45 --traj_int=1 --Vs=10 --record_initpos cth B03 B13 B14 --out_cthmax --Vc=20 --Vd=20 --Td=60 --Vth=5".split()

try:
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
    args = parser.parse_args(test_args)
except:
    args = parser.parse_args()
#%% input file open and select
forward = (args.ward == "forward" or args.ward == "bothward")
backward = (args.ward == "backward" or args.ward == "bothward")
bothward = (args.ward == "bothward")

frames = xr.open_dataset(args.ifn)
tname, yname, xname = frames[args.varname].dims
time_ax_org = frames[tname].values

if args.sector:
    pickup_inds = np.zeros(frames["sector"].shape, bool)
    for sector in args.sector:
        pickup_inds += (frames["sector"] == sector)
    frames = frames.isel({tname:pickup_inds})

if args.itran is None:
    if args.start is None:
        args.start = time_ax_org[0]
    args.start = pd.to_datetime(args.start)
    if args.end is None:
        args.end = time_ax_org[-1]
    args.end = pd.to_datetime(args.end)

    if (args.end - args.start) >= pd.Timedelta("7day"):
        logger.warning("The period lasts more than a week, which may be too long. Please check the period.")

    try:
        args.itran = slice(np.min(np.where(time_ax_org>=args.start)), np.max(np.where(time_ax_org<=args.end)))
    except:
        raise ValueError("Specified time period is out of range")

args.itfst, args.itlst = args.itran.start, args.itran.stop
if backward:
    args.itfst = max([args.itran.start - args.ntrac*args.itstep, 0])
if forward:
    args.itlst = min([args.itran.stop + args.ntrac*args.itstep, time_ax_org.size-1])
frames = frames.isel({tname: slice(args.itfst, args.itlst+1)})
nt = frames[tname].size

if args.maskvar:
    mask = np.zeros(frames[args.maskvar].shape, bool)
    if args.mask_lower_lim:
        mask += frames[args.maskvar] <= args.mask_lower_lim
    if args.mask_upper_lim:
        mask += frames[args.maskvar] >= args.mask_upper_lim

if bothward:
    tg = np.arange(args.ntrac*args.itstep, nt-args.ntrac*args.itstep, args.tidstep)
elif forward:
    tg = np.arange(0, nt-args.ntrac*args.itstep, args.tidstep)
elif backward:
    tg = np.arange(args.ntrac*args.itstep, nt, args.tidstep)
#%% Create coordinates
# Create time coords
base_time = frames[tname].isel({tname:0})
time_ax_values = pd.to_timedelta(frames[tname].isel({tname:tg}) - base_time).total_seconds().values
time_ax = xr.DataArray(time_ax_values, coords={"it":(["it"], tg)}, attrs=dict(long_name="time", units=f'seconds since {base_time.dt.strftime("%F %H:%M:%S").item()}')).rename("time")
it_ax = xr.DataArray(tg, coords={"it":(["it"], tg)}, attrs=dict(long_name="time index", units="")).rename("it")
coords = {"it": it_ax, "time": time_ax}

# Create (x,y) or (r,a) coords
if args.polar is False:
    xg = np.arange(args.xgran.start, args.xgran.stop+args.xint, args.xint)
    xax = xr.DataArray(xg, coords={"x":(["x"], xg)}, attrs=dict(long_name="x", units="km")).rename("x")
    yg = np.arange(args.ygran.start, args.ygran.stop+args.yint, args.yint)
    yax = xr.DataArray(yg, coords={"y":(["y"], yg)}, attrs=dict(long_name="y", units="km")).rename("y")
    xxg, yyg = np.meshgrid(xg, yg)
    axes = [it_ax.name, yax.name, xax.name]
    vshape = [tg.size, yg.size, xg.size]
    initpos_shape = [tg.size, yg.size, xg.size]
    coords.update({"y": yax, "x": xax})
    dim1, dim2, loc1, loc2, v1, v2, ax1, ax2 = yax.name, xax.name, "yloc", "xloc", "vy", "vx", yax, xax
else:
    rg = np.arange(args.rgran.start, args.rgran.stop+args.rint, args.rint)
    rax = xr.DataArray(rg, coords={"r":(["r"], rg)}, attrs=dict(long_name="radius", units="km")).rename("r")
    ag = np.linspace(0, 2*np.pi, args.nath+1)[:-1]
    aax = xr.DataArray(ag, coords={"a":(["a"], ag)}, attrs=dict(long_name="azimuth", units="radian")).rename("a")
    costh = np.cos(ag)
    sinth = np.sin(ag)
    xxg = rax.values[:,None] * costh[None,:]
    yyg = rax.values[:,None] * sinth[None,:]
    axes = [it_ax.name, rax.name, aax.name]
    vshape = [tg.size, rg.size, ag.size]
    initpos_shape = [tg.size, rg.size, ag.size]
    coords.update({"r": rax, "a": aax})
    dim1, dim2, loc1, loc2, v1, v2, ax1, ax2 = rax.name, aax.name, "rloc", "aloc", "vr", "vt", rax, aax

# Create it_rel coords
if args.traj_int is None:
    if args.vagg == "org":
        args.traj_int = 1
    else:
        args.traj_int = args.ntrac

if bothward:
    ntraj = int(2 * (args.ntrac/args.traj_int) + 1) # always odd number (>=1)
    ntraj_half = ntraj//2 # 0, 1, ...
    it_rel = np.arange(-ntraj_half,ntraj_half+1) * args.itstep * args.traj_int # relative time index along the original data
else:
    ntraj = int(args.ntrac/args.traj_int + 1)
    sgn = 1 if forward else -1
    it_rel = np.arange(0,ntraj) * args.itstep * args.traj_int * sgn
it_rel_ax = xr.DataArray(it_rel, coords={"it_rel":(["it_rel"], it_rel)}, attrs=dict(long_name="relative time index along the original data", units="")).rename("it_rel")

axes_t = [axes[0], it_rel_ax.name, *axes[1:]]
vshape_t = [vshape[0], it_rel.size, *vshape[1:]]
coords.update({"it_rel": it_rel_ax})

# Create it_rel_v coords
if args.vagg == "org" and bothward and args.traj_int != 1:
    raise ValueError("When -v=org w/o --forward or --backward, set --traj_int 1")
if bothward:
    ntrajv = ntraj - 1
    it_rel_v = np.arange(-ntraj_half,ntraj_half) * args.itstep * args.traj_int + args.itstep/2
else:
    ntrajv = int((args.ntrac-1)/args.traj_int + 1)
    sgn = 1 if forward else -1
    it_rel_v = (np.arange(0,ntrajv) * args.itstep * args.traj_int + args.itstep/2) * sgn
it_rel_v_ax = xr.DataArray(it_rel_v, coords={"it_rel_v":(["it_rel_v"], it_rel_v)}, attrs=dict(long_name="relative time index for speed", units="")).rename("it_rel_v")
coords.update({"it_rel_v": it_rel_v_ax})

if args.vagg == "org":
    axes = [axes[0], it_rel_v_ax.name, *axes[1:]]
    vshape = [vshape[0], it_rel_v.size, *vshape[1:]]
#%% Create output xarray object | (vx, vy, xloc, yloc, stf, stb) or (vr, vt, rloc, aloc, stf, stb)
fmiss = np.finfo(np.float32).max
ofl = xr.Dataset(
    data_vars={
        v1: (axes, np.full(vshape, np.nan, dtype=np.float32)),
        v2: (axes, np.full(vshape, np.nan, dtype=np.float32)),
        loc1: (axes_t, np.full(vshape_t, np.nan, dtype=np.float32)),
        loc2: (axes_t, np.full(vshape_t, np.nan, dtype=np.float32)),
    },
    coords=coords
)
ofl[v1].attrs.update({"long_name": f"{dim1}-axis velocity", "units":"m/s"})
ofl[v2].attrs.update({"long_name": f"{dim2}-axis velocity", "units":"m/s"})
ofl[loc1].attrs.update({"long_name": f"{dim1} location", "units":ax1.units})
ofl[loc2].attrs.update({"long_name": f"{dim2} location", "units":ax2.units})
ofl["score"] = (["it", "it_rel_v", dim1, dim2], np.full((tg.size, it_rel_v_ax.size, *vshape[-2:]), fmiss, np.float32))
ofl["score"].attrs.update({"long_name": "score", "units":""})

encoding = {}
encoding.update({key: {"_FillValue": fmiss} for key in ofl.data_vars.keys()})

if forward:
    ofl = ofl.assign({"stf": ([it_ax.name, dim1, dim2], np.full([tg.size, ax1.size, ax2.size], -10, dtype=np.int16))})
    ofl["stf"].attrs.update({"long_name": "forward tracking status", "units":"", "flags": [0,1,2,3,4,5,6,7,8,9,10,11], "flag means": ["Alive", "Invalid time index", "Invalid sub image", "Low contrast", "Side zsub peak", "Invalid time index", "Can't get score", "Can't get score peak", "Low score", "Large V change", "Exceed V limit or Large back-forward change", "Max dt"]})
    encoding.update({"stf": {"_FillValue": -10}})
if backward:
    ofl = ofl.assign({"stb": ([it_ax.name, dim1, dim2], np.full([tg.size, ax1.size, ax2.size], -10, dtype=np.int16))})
    ofl["stb"].attrs.update({"long_name": "backward tracking status", "units":"", "flags": [0,1,2,3,4,5,6,7,8,9,10,11], "flag means": ["Alive", "Invalid time index", "Invalid sub image", "Low contrast", "Side zsub peak", "Invalid time index", "Can't get score", "Can't get score peak", "Low score", "Large V change", "Exceed V limit or Large back-forward change", "Max dt"]})
    encoding.update({"stb": {"_FillValue": -10}})

if bothward:
    ofl["vxfm"] = (axes, np.full(vshape, np.nan, dtype=np.float32))
    ofl["vyfm"] = (axes, np.full(vshape, np.nan, dtype=np.float32))
    ofl["vxbm"] = (axes, np.full(vshape, np.nan, dtype=np.float32))
    ofl["vybm"] = (axes, np.full(vshape, np.nan, dtype=np.float32))

# Compute `time2` (2-D time array along [it, it_rel] axes)
it_plus_it_rel = (it_ax + it_rel_ax).values
time2 = frames[tname].isel({tname:it_plus_it_rel.ravel()}).values.reshape(it_plus_it_rel.shape)
ofl["time2"] = (["it", "it_rel"], time2)

# Initialize variables for recording at initial template positions
if args.record_initpos:
    for varname in args.record_initpos:
        ofl[varname] = (axes, np.zeros(initpos_shape, dtype=np.float32))
        ofl[varname].attrs.update(frames[varname].attrs)

# Initialize variables for recording along trajectory
if args.record_alongtraj:
    for varname in args.record_alongtraj:
        ofl[varname] = (axes_t, np.zeros(vshape_t, dtype=np.float32))
        ofl[varname].attrs.update(frames[varname].attrs)

# Record execution command & history into attributes
args_str = ""
for key, val in vars(args).items():
    args_str += f' --{key}={val}'
ofl.attrs["exec"] = f'python {__file__}' + args_str
ofl.attrs.update(vars(args))
ofl.attrs.update({"history": f'{os.getenv("USER")} {pd.Timestamp.now().strftime("%F %H:%M:%S UTC")}'})
#%% Setup VTTrac object
z_values = frames[args.varname].astype(np.float32).values
zmiss = None
if np.isnan(z_values).any() or args.revrot != 0.0:
    zmiss = np.finfo(z_values.dtype).max
    z_values = np.nan_to_num(z_values, nan=zmiss)
t = (frames[tname] - frames[tname][0]).dt.seconds.values.astype(np.float64)
tunit = "s"

if args.maskvar:
    vtt = VTTrac.VTT(z_values, t, zmiss=zmiss, fmiss=fmiss, mask=mask.values)
    del mask
else:
    vtt = VTTrac.VTT(z_values, t, zmiss=zmiss, fmiss=fmiss)

# VTT dimension setup
xcoord = frames[args.varname].coords[xname]
x0 = xcoord.min().item()
dx = abs(((xcoord[-1]-xcoord[0])/(xcoord.size-1)).item())
ycoord = frames[args.varname].coords[yname]
y0 = ycoord.min().item()
dy = abs(((ycoord[-1]-ycoord[0])/(ycoord.size-1)).item())

if xcoord.attrs["units"] == "km":
    ucfact, ucufact = 1e3, "m/km"
else:
    ucfact, ucufact = 1, None
vtt.set_grid_par(x0, y0, dx, dy, ucfact, ucufact)

# VTT tracking setup
args.nsx = args.nsx or args.ns
args.nsy = args.nsy or args.ns
setup_kwargs = dict(
    nsx=args.nsx, nsy=args.nsy, ntrac=args.ntrac,
    Sth0=args.Sth0, Sth1=args.Sth1, subgrid=not args.no_subgrid,
    Cth=args.Cth, peak_inside_th=args.peak_inside_th,
    itstep=args.itstep, vxch=args.Vc, vych=args.Vc, 
    use_init_temp=args.use_init_temp, min_samples=args.min_samples)
if args.hs:
    setup_kwargs["ixhw"] = args.hs
    setup_kwargs["iyhw"] = args.hs
else:
    setup_kwargs["vxhw"] = args.Vs
    setup_kwargs["vyhw"] = args.Vs
vtt.setup_eq_grid(**setup_kwargs)

tdiff = np.diff(t)
dtmax = np.max(tdiff[tdiff<=args.dtlimit])
vtt["ixhw"], vtt["iyhw"] = vtt.calc_ixyhw_from_v_eq_grid(args.Vs, args.Vs, dtmax)

# Output setup
ofl.attrs.update(vtt.attrs)
if args.out_subimage:
    zss_shape = (tg.size, it_rel_ax.size, vtt.o.nsy, vtt.o.nsx, *vshape[-2:])
    ofl["zss"] = (["it", "it_rel", "sy", "sx", dim1, dim2], np.full(zss_shape, np.nan, np.float32))
    ofl["zss"].attrs.update({"long_name": "sub image", "units":frames[args.varname].attrs["units"], "_FillValue": vtt.zmiss})
if args.out_score_ary:
    score_ary_shape = (tg.size, it_rel_v_ax.size, 2*vtt.o.iyhw+1, 2*vtt.o.ixhw+1, *vshape[-2:])
    ofl["score_ary"] = (["it", "it_rel_v", "scy", "scx", dim1, dim2], np.full(score_ary_shape, np.nan, np.float32))
    ofl["score_ary"].attrs.update({"long_name": "score array", "units":"", "_FillValue": vtt.fmiss})

#%% Set pickup it_rel and it_rel_v index
if bothward:
    pickup_it_rel_v = np.arange(ntraj_half) * args.traj_int
    pickup_it_rel = np.arange(ntraj_half+1) * args.traj_int
else:
    pickup_it_rel_v = np.arange(ntrajv) * args.traj_int
    pickup_it_rel = np.arange(ntraj) * args.traj_int

if args.revrot:
    it_rel_without_0 = it_rel[it_rel!=0]
    deg_per_sec = np.rad2deg(args.revrot)

dt_rel = (ofl["time2"]-ofl["time2"].sel(it_rel=0)).data.astype('timedelta64[s]').astype(float)
maxdts = np.max(np.abs(dt_rel), axis=1)
#%% Perform tracking
for j, tid0 in enumerate(tg.tolist()):
    # if j != 5: # for debug
    #     continue
    if (j%10) == 0 or j == tg.size-1:
        logger.info(f"Processing: {j+1}/{tg.size}")
    if maxdts[j] >= args.dtlimit:
        continue
    if args.revrot:
        vtt.o.z[tid0,:,:] = z_values[tid0]
        for it_rel_i in it_rel_without_0:
            img = Image.fromarray(z_values[tid0+it_rel_i])
            dt = t[tid0+it_rel_i] - t[tid0]
            vtt.o.z[tid0+it_rel_i,:,:] = np.array(img.rotate(deg_per_sec*dt, resample=Image.BICUBIC, fillcolor=zmiss))
    if forward:
        vtt["itstep"] = abs(vtt.itstep)
        vtt["ixhw"], vtt["iyhw"] = vtt.calc_ixyhw_from_v_eq_grid(args.Vs, args.Vs, dtmax)
        fward_res = vtt.trac_eq_grid(tid0, xxg, yyg, out_subimage=args.out_subimage, out_score_ary=args.out_score_ary)
    if backward:
        vtt["itstep"] = -abs(vtt.itstep)
        vtt["ixhw"], vtt["iyhw"] = vtt.calc_ixyhw_from_v_eq_grid(args.Vs, args.Vs, dtmax)
        bward_res = vtt.trac_eq_grid(tid0, xxg, yyg, out_subimage=args.out_subimage, out_score_ary=args.out_score_ary)
    # break
    if args.vagg == "org":
        if bothward:
            vx = xr.concat([
                    bward_res["vx"].isel(it_rel_v=np.flip(pickup_it_rel_v)),
                    fward_res["vx"].isel(it_rel_v=pickup_it_rel_v)
                ], dim="it_rel_v")
            vy = xr.concat([
                    bward_res["vy"].isel(it_rel_v=np.flip(pickup_it_rel_v)),
                    fward_res["vy"].isel(it_rel_v=pickup_it_rel_v)
                ], dim="it_rel_v")
        elif forward:
            vx = fward_res["vx"].isel(it_rel_v=pickup_it_rel_v)
            vy = fward_res["vy"].isel(it_rel_v=pickup_it_rel_v)
        else:
            vx = bward_res["vx"].isel(it_rel_v=pickup_it_rel_v)
            vy = bward_res["vy"].isel(it_rel_v=pickup_it_rel_v)
    else:
        if args.vagg == "mean":
            if forward:
                vxfm = fward_res["vx"].mean(axis=0, skipna=True)
                vyfm = fward_res["vy"].mean(axis=0, skipna=True)
            if backward:
                vxbm = bward_res["vx"].mean(axis=0, skipna=True)
                vybm = bward_res["vy"].mean(axis=0, skipna=True)
        elif args.vagg == "startend":
            if forward:
                start_end_dtf = fward_res["t"][tid0+fward_res["it_rel"]].values.ptp()
                vxfm = (fward_res["xloc"].isel(it_rel=-1) - fward_res["xloc"].isel(it_rel=0))*vtt.ucfact/start_end_dtf
                vyfm = (fward_res["yloc"].isel(it_rel=-1) - fward_res["yloc"].isel(it_rel=0))*vtt.ucfact/start_end_dtf
            if backward:
                start_end_dtb = bward_res["t"][tid0+bward_res["it_rel"]].values.ptp()
                vxbm = (bward_res["xloc"].isel(it_rel=0) - bward_res["xloc"].isel(it_rel=-1))*vtt.ucfact/start_end_dtb
                vybm = (bward_res["yloc"].isel(it_rel=0) - bward_res["yloc"].isel(it_rel=-1))*vtt.ucfact/start_end_dtb
        if bothward:
            vx = (vxfm+vxbm)/2
            vy = (vyfm+vybm)/2
            ofl["vxfm"].data[j] = vxfm.data
            ofl["vyfm"].data[j] = vyfm.data
            ofl["vxbm"].data[j] = vxbm.data
            ofl["vybm"].data[j] = vybm.data
        elif forward:
            vx, vy = vxfm, vyfm
        elif backward:
            vx, vy = vxbm, vybm
    
    if bothward:
        xtraj = xr.concat([
            bward_res["xloc"].isel(it_rel=np.flip(pickup_it_rel[1:])),
            fward_res["xloc"].isel(it_rel=pickup_it_rel)
        ], dim="it_rel")
        ytraj = xr.concat([
            bward_res["yloc"].isel(it_rel=np.flip(pickup_it_rel[1:])),
            fward_res["yloc"].isel(it_rel=pickup_it_rel)
        ], dim="it_rel")
        ofl["score"].data[j] = xr.concat([
            bward_res["score"].isel(it_rel_v=np.flip(pickup_it_rel_v)),
            fward_res["score"].isel(it_rel_v=pickup_it_rel_v)
        ], dim="it_rel_v")
        ofl["stf"].data[j] = fward_res["status"].data
        ofl["stb"].data[j] = bward_res["status"].data
    elif forward:
        xtraj = fward_res["xloc"].isel(it_rel=pickup_it_rel)
        ytraj = fward_res["yloc"].isel(it_rel=pickup_it_rel)
        ofl["score"].data[j] = fward_res["score"].isel(it_rel_v=pickup_it_rel_v)
        ofl["stf"].data[j] = fward_res["status"].data
    elif backward:
        xtraj = bward_res["xloc"].isel(it_rel=pickup_it_rel)
        ytraj = bward_res["yloc"].isel(it_rel=pickup_it_rel)
        ofl["score"].data[j] = bward_res["score"].isel(it_rel_v=pickup_it_rel_v)
        ofl["stb"].data[j] = bward_res["status"].data

    if not args.polar:
        ofl["vx"].data[j] = vx.data
        ofl["vy"].data[j] = vy.data
        ofl["xloc"].data[j] = xtraj.data
        ofl["yloc"].data[j] = ytraj.data
    else:
        vr = vx*costh[None,:] + vy*sinth[None,:]
        vt = -vx*sinth[None,:] + vy*costh[None,:]
        ofl["vr"].data[j] = vr.data
        ofl["vt"].data[j] = vt.data
        rtraj = np.hypot(xtraj, ytraj)
        atraj = np.arctan2(ytraj, xtraj)
        ofl["rloc"].data[j] = rtraj.data
        ofl["aloc"].data[j] = atraj.data
    
    if args.out_subimage:
        if bothward:
            zss = xr.concat([
                bward_res["zss"].isel(it_rel=np.flip(pickup_it_rel[1:])),
                fward_res["zss"].isel(it_rel=pickup_it_rel)
            ], dim="it_rel")
        elif forward:
            zss = fward_res["zss"].isel(it_rel=pickup_it_rel)
        elif backward:
            zss = bward_res["zss"].isel(it_rel=pickup_it_rel)
        ofl["zss"].data[j] = zss.data
        
    if args.out_score_ary:
        if bothward:
            score_ary = xr.concat([
                bward_res["score_ary"].isel(it_rel_v=np.flip(pickup_it_rel_v)),
                fward_res["score_ary"].isel(it_rel_v=pickup_it_rel_v)
            ], dim="it_rel_v")
        elif forward:
            score_ary = fward_res["score_ary"].isel(it_rel_v=pickup_it_rel_v)
        elif backward:
            score_ary = bward_res["score_ary"].isel(it_rel_v=pickup_it_rel_v)
        pad_width = (ofl.scx.size - score_ary.scx.size)//2
        if pad_width == 0:
            ofl["score_ary"].data[j] = score_ary.data
        else:
            ofl["score_ary"].data[j] = np.pad(score_ary.data, [(0,0),(pad_width,pad_width),(pad_width,pad_width),(0,0),(0,0)], constant_values=np.nan)
    # if j == 5: # for debug
    #     break
#%% Process revrot
if args.revrot:
    if args.polar:
        r2d = ofl["r"].values[:,None].repeat(ofl["a"].size, axis=1)
    else:
        a2d = np.arctan2(ofl["y"].T, ofl["x"])
        r2d = np.hypot(ofl["y"].T, ofl["x"]).values
    revrot_mps = args.revrot*r2d*1000
    azimuth_displasement_on_it_rel = dt_rel * args.revrot
    if args.polar:
        ofl["vt"] += revrot_mps
        ofl["aloc"] = (ofl["aloc"] + azimuth_displasement_on_it_rel[:,:,None,None]) % (2*np.pi)
    else:
        u_rot = -np.sin(a2d) * revrot_mps
        v_rot = np.cos(a2d) * revrot_mps
        ofl["vx"] += u_rot
        ofl["vy"] += v_rot
        aloc = np.arctan2(ofl["yloc"], ofl["xloc"]).values
        aloc += azimuth_displasement_on_it_rel[:,:,None,None]
        rloc = np.hypot(ofl["yloc"], ofl["xloc"]).values
        ofl["xloc"].data = np.cos(aloc) * rloc
        ofl["yloc"].data = np.sin(aloc) * rloc
    
    if bothward:
        if args.polar:
            x = ofl["r"] * np.cos(ofl["a"])
            y = ofl["r"] * np.sin(ofl["a"])
            a2d = np.arctan2(y, x)
        u_rot = -np.sin(a2d) * revrot_mps
        v_rot = np.cos(a2d) * revrot_mps
        ofl["vxfm"] += u_rot
        ofl["vyfm"] += v_rot
        ofl["vxbm"] += u_rot
        ofl["vybm"] += v_rot
#%% Perform wind-speed screening
valid_tyx = np.zeros(vshape, bool)
if args.vlim > 0: # limiting speed
    vabs = np.hypot(ofl[v1],ofl[v2])
    valid_tyx = vabs <= args.vlim

if bothward: # limiting differences in backward & forward tracking
    valid_tyx += (np.hypot(ofl["vxfm"]-ofl["vxbm"], ofl["vyfm"]-ofl["vybm"]) <= args.Vd**2)

    if args.Td is not None:
        dot_product = ofl["vxfm"]*ofl["vxbm"] + ofl["vyfm"]*ofl["vybm"]
        vabsf = np.hypot(ofl["vxfm"], ofl["vyfm"])
        vabsb = np.hypot(ofl["vxbm"], ofl["vybm"])
        angle_diff = np.arccos(dot_product/vabsf/vabsb)
        
        if args.Vth > 0:
            valid_tyx += ~((angle_diff > np.deg2rad(args.Td)) * ((vabsf >= args.Vth) + (vabsb >= args.Vth)))
        else:
            valid_tyx += ~(angle_diff > np.deg2rad(args.Td))
ofl = ofl.drop_vars(["vxfm","vyfm","vxbm","vybm"])

ofl[[v1,v2]] = ofl[[v1,v2]].where(valid_tyx)
if forward:
    ofl["stf"] = ofl["stf"].where(valid_tyx * (ofl["stf"]==0), 10)
if backward:
    ofl["stb"] = ofl["stb"].where(valid_tyx * (ofl["stb"]==0), 10)

if args.vagg in ("mean","startend"):
    valid_tityx = valid_tyx.expand_dims({"it_rel": ofl.it_rel}, axis=0)
    ofl[[loc1,loc2]] = ofl[[loc1,loc2]].where(valid_tityx)
#%% initpos, alongtraj, PSR

if args.record_initpos:
    if args.polar:
        ofl["x"] = ofl.r * np.cos(ofl.a)
        ofl["y"] = ofl.r * np.sin(ofl.a)
    for varname in args.record_initpos:
        ofl[varname].data = frames[varname].interp({tname:frames[tname][tg], yname:ofl["y"], xname:ofl["x"]})
    if args.polar:
        ofl = ofl.drop(["x","y"])

if args.record_alongtraj or args.out_cthmax or args.out_cthmax:
    if args.polar:
        ofl["xloc"] = ofl["rloc"] * np.cos(ofl["aloc"])
        ofl["yloc"] = ofl["rloc"] * np.sin(ofl["aloc"])
    times_on_traj = ofl["time2"].data[:,:,None,None].repeat(vshape[1], axis=2).repeat(vshape[2], axis=3)
    times_on_traj_1d = xr.DataArray(times_on_traj.ravel(), dims="_")
    xlocs, ylocs = ofl["xloc"].copy(deep=True), ofl["yloc"].copy(deep=True)
    xlocs.sel(it_rel=0)[:] = xr.DataArray(xxg, dims=["y","x"]) # fill initial x position regardless of tracking status
    ylocs.sel(it_rel=0)[:] = xr.DataArray(yyg, dims=["y","x"]) # fill initial y position regardless of tracking status
    xlocs_1d = xr.DataArray(xlocs.data.ravel(), dims="_")
    ylocs_1d = xr.DataArray(ylocs.data.ravel(), dims="_")

    if args.record_alongtraj:
        for varname in args.record_alongtraj:
            ofl[varname].data = frames[varname].interp({tname:times_on_traj_1d, yname:ylocs_1d, xname:xlocs_1d}).data.reshape(vshape_t)

    if args.out_cthmax or args.out_cthmax:
        cth_alongtraj = frames[args.cth].interp({tname:times_on_traj_1d, yname:ylocs_1d, xname:xlocs_1d}).data.reshape(vshape_t)
        if args.out_cthmin:
            ofl[f"{args.cth}min"] = xr.DataArray(cth_alongtraj, dims=axes_t, coords=ofl.xloc.coords).min("it_rel")
        if args.out_cthmax:
            ofl[f"{args.cth}max"] = xr.DataArray(cth_alongtraj, dims=axes_t, coords=ofl.xloc.coords).max("it_rel")
    if args.polar:
        ofl = ofl.drop(["xloc","yloc"])

if args.out_score_ary and args.out_psr:
    def get_PSR(flows, around_ratio=0.15):
        """Peak-to-Sidelobe Ratio"""
        max_scores = flows["score_ary"].max(dim=["scx","scy"])
        max_is_true = (flows["score_ary"] == max_scores)

        around_wh = int((np.sqrt(around_ratio*(flows["scx"].size*flows["scy"].size))).round())
        dilated = ndi.binary_dilation(max_is_true, structure=np.ones([around_wh,around_wh], bool).reshape(1,1,around_wh,around_wh,1,1))
        sidelobes = xr.where(~dilated, flows["score_ary"], np.nan)

        PSRs = (flows["score"] - sidelobes.mean())/sidelobes.std()
        return PSRs.rename("psr")
    around_ratio = 0.15
    ofl["psr"] = get_PSR(ofl, around_ratio)
    ofl["psr"].attrs.update({"long_name": "peak-to-sidelobe ratio", "around_ratio": around_ratio, "units": ""})
#%% output
for key, value in ofl.attrs.items():
    if value is None:
        ofl.attrs[key] = "None"
    elif isinstance(value, pd.Timestamp):
        ofl.attrs[key] = value.strftime("%F %H:%M:%S")
    elif isinstance(value, slice):
        ofl.attrs[key] = [value.start, value.stop]
    elif isinstance(value, bool):
        ofl.attrs[key] = int(value)

for var in ofl:
    encoding[var] = {"complevel": args.complevel, "zlib": True}
ofl.to_netcdf(args.ofn, encoding=encoding)
logger.info(f"[SUCCESS] {args.ofn}")
# %%
