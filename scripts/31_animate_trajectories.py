#%%
"""
# openTCAMV -- 31_animate_trajectories.py
Animate the tracked cloud parcels behind the AMVs: every template moves along its trajectory over the images it was
tracked in, leaving a trail that thins with age -- the most direct check that the tracking followed the clouds.

One cycle per AMV time: the parcels of the AMVs centered on an image move through the 2 ntrac + 1 images of their
trajectories, then one image is shown alone and the next cycle starts, so every image appears exactly once.
`--corotate` draws each cycle in a frame rotating with the vortex, where the parcels' motion relative to the rotation
(what `--revrot` tracks) is easier to follow. The image is shown standardized on a gray scale that brings out the cloud
texture; `--raw` shows its values on a colorbar instead, e.g. infrared brightness temperatures with `--image_cmap batlow`.

The drawing is `opentcamv.trajectory_movie.animate_trajectories`, a self-contained function (numpy + matplotlib only)
that can be copied into any workflow; this script only reads openTCAMV files into it.

## Data requirement
AMV file: output of `10_conduct_tracking.py` or `20_finalize_tracking.py`, with `xloc`/`yloc` (`rloc`/`aloc` under
`--polar`) and `time2`. Track with `--traj_int=1` so that every image along a trajectory is recorded.
Image file: the NetCDF that `10_` tracked (default: the AMV file's `ifn` attribute).

## Usage
python 31_animate_trajectories.py ../sample/2017_Lan_ns7_nt1.nc --log --contrast 0.5 --zoom 0 0 15 --rings 10 20
python 31_animate_trajectories.py AMVs_B13.nc --raw --image_cmap batlow --vlim 190 300 --zoom 0 0 15 --zoom_vlim 270 300

## Reference
Tsukada, T., Horinouchi, T., & Tsujino, S. (2024). Wind distribution in the eye of tropical cyclone revealed by a novel atmospheric motion vector derivation. Journal of Geophysical Research: Atmospheres, 129, e2023JD040585. https://doi.org/10.1029/2023JD040585
"""

import os
import logging
import argparse
from functools import partial

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")

from opentcamv.trajectory_movie import animate_trajectories, local_contrast, match_frames

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s %(name)s %(lineno)d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

parser = argparse.ArgumentParser(description="Animate the tracked parcels (AMV trajectories) over the images they were tracked in")
parser.add_argument("ifn", help="AMV file from 10_conduct_tracking.py or 20_finalize_tracking.py")
parser.add_argument("--image", help="image NetCDF the AMVs were tracked in (default: the AMV file's `ifn` attribute)")
parser.add_argument("--varname", help="image variable (default: the AMV file's `varname` attribute)")
parser.add_argument("-o", "--ofn", help="output movie, .mp4 (needs ffmpeg) or .gif (default: <AMV file dir>/outputs/<AMV file name>_trajectories.mp4)")
parser.add_argument("-s", "--start", help="first image to show, e.g. 2017-10-21T02:30")
parser.add_argument("-e", "--end", help="last image to show")
parser.add_argument("--omega", type=float, help="Ω_i candidate to show from a 10_ output holding several (`omega` dimension)")
parser.add_argument("--xlim", type=float, nargs=2, help="x range of the main panel (km; default: the AMV grid)")
parser.add_argument("--ylim", type=float, nargs=2, help="y range of the main panel (km; default: the AMV grid)")
parser.add_argument("--stride", type=int, default=2, help="draw every N-th AMV grid point in the main panel")
parser.add_argument("--checker", action="store_true", help="of those, keep every other one in a checkerboard (spacing x sqrt 2)")
parser.add_argument("--zoom", type=float, nargs=3, action="append", metavar=("CX", "CY", "HALF"), help="add a zoom panel centered at (CX, CY) with half-width HALF (km); repeatable")
parser.add_argument("--zoom_stride", type=int, default=1, help="--stride of the zoom panels")
parser.add_argument("--zoom_checker", action="store_true", help="--checker of the zoom panels")
parser.add_argument("--titles", nargs="+", help="panel titles: the main panel's, then the zoom panels'")
parser.add_argument("--corotate", type=float, default=0.0, help="draw each cycle in a frame rotating counterclockwise at this angular velocity about the center, aligned with the storm-centered frame at the AMV time (rad/s; 0 = storm-centered)")
parser.add_argument("--color", default="yellow", help="parcel color, or 'speed' to color by the AMV speed (storm-relative, or relative to the --corotate frame)")
parser.add_argument("--clim", type=float, nargs=2, help="speed range of --color speed (m/s)")
parser.add_argument("--head_color", help="fill of the parcel dots: a color, or 'image' for the image color where the parcel first appears (the start of its trajectory); the trails keep --color (default: --color)")
parser.add_argument("--size_scale", type=float, default=1.0, help="scale of the parcel dots' diameter (their area scales by its square)")
parser.add_argument("--linewidth", type=float, default=2.2, help="trail width at the parcel (pt; 0.75x in the main panel when zoom panels are added)")
parser.add_argument("--tail_width", type=float, default=0.1, help="trail width at the start of the trajectory, as a fraction of --linewidth")
parser.add_argument("--log", action="store_true", help="show the log of the image (visible reflectance)")
parser.add_argument("--contrast", type=float, default=0.0, help="fraction of the local mean (over --contrast_fwhm) removed from the image; 0.5 brings out the cloud texture, 0 = off")
parser.add_argument("--contrast_fwhm", type=float, default=8.0, help="width of the local mean for --contrast (km)")
parser.add_argument("--raw", action="store_true", help="show the image values themselves on a colorbar (e.g. brightness temperature) instead of standardizing them")
parser.add_argument("--image_cmap", default="gray", help="colormap of the image: a matplotlib name, or one of Crameri's Scientific colour maps such as batlow (needs cmcrameri)")
parser.add_argument("--image_label", help="label of the --raw colorbar (default: the variable's long_name and units)")
parser.add_argument("--vlim", type=float, nargs=2, help="color limits of the image: of the standardized image (default -2.4 2.4), or in the image's units under --raw (default: the first image's range over the main panel)")
parser.add_argument("--zoom_vlim", type=float, nargs=2, help="--vlim of the zoom panels, each then with its own colorbar under --raw (default: --vlim)")
parser.add_argument("--rings", type=float, nargs="*", default=[], help="radii of range rings about the center (km)")
parser.add_argument("--rest", type=int, default=1, help="images shown alone after each cycle")
parser.add_argument("--fps", type=float, default=2.0, help="movie frames per second")
parser.add_argument("--dpi", type=float, default=110, help="movie resolution (dots per inch)")
parser.add_argument("--max_frames", type=int, help="only the first N movie frames (quick test)")
parser.add_argument("--title", help="frame-label prefix (default: the image file's `name` attribute and the variable)")
args = parser.parse_args()


def colormap(name):
    """`name` as matplotlib registers it; cmcrameri registers Crameri's Scientific colour maps as `cmc.<name>`."""
    if name not in matplotlib.colormaps:
        try:
            import cmcrameri.cm  # noqa: F401
        except ImportError:
            pass
    for registered in (name, f"cmc.{name}"):
        if registered in matplotlib.colormaps:
            return registered
    parser.error(f"unknown colormap {name!r} (Crameri's maps such as batlow need `pip install cmcrameri`)")


image_cmap = colormap(args.image_cmap)

# %% trajectories
A = xr.open_dataset(args.ifn)
if "omega" in A.dims:
    if args.omega is None and A.sizes["omega"] > 1:
        parser.error(f"{args.ifn} holds {A.sizes['omega']} Ω_i candidates; pick one with --omega (or finalize with 20_)")
    A = A.isel(omega=0) if args.omega is None else A.sel(omega=args.omega, method="nearest")
polar = "rloc" in A
locs = ("rloc", "aloc") if polar else ("xloc", "yloc")
seed_dims = [d for d in A[locs[0]].dims if d not in ("it", "it_rel")]
order = np.argsort(A.it_rel.values)                       # steps in time order (backward tracking stores them reversed)
L1, L2 = (A[name].transpose("it", "it_rel", *seed_dims).values[:, order] for name in locs)
xloc, yloc = (L1 * np.cos(L2), L1 * np.sin(L2)) if polar else (L1, L2)
valid = A["vt" if polar else "vx"].notnull()
if "it_rel_v" in valid.dims:                              # --vagg=org
    valid = valid.all("it_rel_v")
valid = valid.transpose("it", *seed_dims).values
time2 = A.time2.transpose("it", "it_rel").values[:, order]
ref_step = int(np.flatnonzero(A.it_rel.values[order] == 0)[0])

# %% images
image = args.image or A.attrs.get("ifn")
if image and not os.path.exists(image):                   # `ifn` is relative to where 10_ ran
    beside = os.path.join(os.path.dirname(args.ifn), os.path.basename(image))
    image = beside if os.path.exists(beside) else image
if not image or not os.path.exists(image):
    parser.error(f"image file {image!r} not found; pass --image")
I = xr.open_dataset(image)
varname = args.varname or A.attrs.get("varname")
Z = I[varname]
tname, yname, xname = Z.dims
times = Z[tname].values
shown = np.ones(times.size, bool)
if args.start:
    shown &= times >= pd.Timestamp(args.start).to_datetime64()
if args.end:
    shown &= times <= pd.Timestamp(args.end).to_datetime64()
Z, times = Z.isel({tname: shown}), times[shown]
frame_index = match_frames(time2, times)
logger.info(f"{times.size} images {times[0]} - {times[-1]}; {A.it.size} AMV times, {(frame_index >= 0).all(1).sum()} with every step on an image")

# %% panels
if polar:
    rmax = float(A.r.max())
    extent = ((-rmax, rmax), (-rmax, rmax))
    grid_step = None
else:
    extent = ((float(A.x.min()), float(A.x.max())), (float(A.y.min()), float(A.y.max())))
    grid_step = float(abs(A.x[1] - A.x[0]))
zooms = args.zoom or []
titles = list(args.titles or []) + [""] * (1 + len(zooms))


def panel_title(name, stride, checker):
    if grid_step is None:
        return name
    spacing = f"parcels every {stride * grid_step * (np.sqrt(2) if checker else 1):.1f} km"
    return f"{name}, {spacing}" if name else spacing[0].upper() + spacing[1:]


panels = [dict(xlim=args.xlim or extent[0], ylim=args.ylim or extent[1], stride=args.stride, checker=args.checker,
               linewidth=args.linewidth * (0.75 if zooms else 1.0), size=(9 if zooms else 14) * args.size_scale ** 2,
               title=panel_title(titles[0], args.stride, args.checker))]
for n, (cx, cy, half) in enumerate(zooms, start=1):
    panels.append(dict(center=(cx, cy), half=half, stride=args.zoom_stride, checker=args.zoom_checker, linewidth=args.linewidth,
                       size=16 * args.size_scale ** 2, title=panel_title(titles[n], args.zoom_stride, args.zoom_checker), vlim=args.zoom_vlim))

# %% movie
dx = float(abs(I[xname][1] - I[xname][0]))
transform = partial(local_contrast, dx=dx, fwhm=args.contrast_fwhm, fraction=args.contrast, log=args.log) if (args.log or args.contrast) else None
title = args.title if args.title is not None else " ".join(str(v) for v in (I.attrs.get("name"), varname) if v)
image_label = None
if args.raw:
    units = Z.attrs.get("units")
    image_label = args.image_label if args.image_label is not None else Z.attrs.get("long_name", varname) + (f" ({units})" if units else "")
movie = animate_trajectories(
    Z, I[xname].values, I[yname].values, xloc, yloc, frame_index, times=times, valid=valid, panels=panels,
    color=args.color, clim=args.clim, head_color=args.head_color, linewidth=args.linewidth, tail_width=args.tail_width, omega=args.corotate,
    center=(0.0, 0.0), ref_step=ref_step, rings=args.rings, image_transform=transform, standardize=not args.raw,
    vlim=args.vlim, image_cmap=image_cmap, image_label=image_label, rest=args.rest, title=title,
)
if args.max_frames:
    movie.schedule = movie.schedule[:args.max_frames]
ncycle = len({j for _, j, _ in movie.schedule if j is not None})
logger.info(f"{ncycle} trajectory cycles, {len(movie.schedule)} movie frames")

ofn = args.ofn or os.path.join(os.path.dirname(os.path.abspath(args.ifn)), "outputs", os.path.splitext(os.path.basename(args.ifn))[0] + "_trajectories.mp4")
os.makedirs(os.path.dirname(os.path.abspath(ofn)), exist_ok=True)
movie.save(ofn, fps=args.fps, dpi=args.dpi)
logger.info(f"[Saved] see {ofn}")
# %%
