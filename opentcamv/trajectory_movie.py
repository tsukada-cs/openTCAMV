"""Animate tracked cloud parcels over the images they were tracked in.

An AMV is the mean motion of a template followed through a few consecutive images. Moving every template along its
trajectory, on the images themselves, shows directly whether the tracking followed the clouds. This module makes that
animation for any tracker: give it the image sequence, the parcel positions along every trajectory, and the image index
of every trajectory step (`match_frames` derives it from the step times).

The file is self-contained -- numpy and matplotlib only (scipy for `local_contrast` only), nothing from openTCAMV -- so
it can be copied into any project as is. `scripts/31_animate_trajectories.py` is openTCAMV's front end to it.

Example (positions and pixel coordinates in km, `frame_index` from `match_frames`)::

    movie = animate_trajectories(images, x, y, xloc, yloc, frame_index, times=times, center=(0, 0))
    movie.save("trajectories.mp4")      # or .gif; movie.animation() for Jupyter
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm, colors, transforms
from matplotlib.collections import LineCollection

__all__ = ["animate_trajectories", "TrajectoryMovie", "movie_schedule", "match_frames", "local_contrast"]


@dataclass
class TrajectoryMovie:
    """The figure `animate_trajectories` builds, and how to play it.

    `draw(i)` renders movie frame `i` in place (`fig.savefig` then saves that frame); `schedule[i]` is that frame's
    `(image index, trajectory set, step)`, the last two None while an image is shown alone; `mappable` feeds a colorbar
    when the parcels are colored by a value.
    """

    fig: plt.Figure
    axes: list
    schedule: list
    draw: Callable[[int], list]
    mappable: Optional[cm.ScalarMappable] = None

    def animation(self, fps: float = 2.0) -> animation.FuncAnimation:
        """A `FuncAnimation` over the whole schedule (in Jupyter: `HTML(movie.animation().to_jshtml())`)."""
        return animation.FuncAnimation(self.fig, self.draw, frames=len(self.schedule), interval=1000.0 / fps, blit=False)

    def save(self, path, fps: float = 2.0, dpi: float = 110) -> None:
        """Write the movie: `.mp4` through ffmpeg (H.264, yuv420p, even frame size: plays in browsers and slide
        software), `.gif` through Pillow (no ffmpeg needed), any other extension through matplotlib's default writer."""
        path = str(path)
        if path.endswith(".mp4"):
            if not animation.FFMpegWriter.isAvailable():
                raise RuntimeError("ffmpeg not found: install it (e.g. `conda install ffmpeg`), or save as .gif")
            writer = animation.FFMpegWriter(fps=fps, codec="libx264", extra_args=[
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow",
                "-movflags", "+faststart"])
        elif path.endswith(".gif"):
            writer = animation.PillowWriter(fps=fps)
        else:
            self.animation(fps).save(path, fps=fps, dpi=dpi)
            return
        self.animation(fps).save(path, writer=writer, dpi=dpi)


def movie_schedule(frame_index: np.ndarray, usable: np.ndarray, n_images: int, rest: int = 1) -> list:
    """The movie's frames, as `(image index, trajectory set, step)`.

    Walks through the images once, in order. Where a usable set starts, its cycle plays: every image from its first
    step to its last, each with the latest step at or before it, then `rest` images alone; any other image is shown
    alone (set and step None). With a trajectory set centered on every image (openTCAMV's bothward tracking), one cycle
    plays every `2 ntrac + 1 + rest` images and every image appears exactly once.
    """
    first = {}
    for j in np.flatnonzero(usable)[::-1]:                  # the earliest set wins where several start together
        first[int(frame_index[j, 0])] = int(j)
    out, k = [], 0
    while k < n_images:
        j = first.get(k)
        if j is None:
            out.append((k, None, None))
            k += 1
            continue
        f = frame_index[j]
        out += [(i, j, int(np.searchsorted(f, i, side="right")) - 1) for i in range(f[0], f[-1] + 1)]
        out += [(i, None, None) for i in range(f[-1] + 1, min(f[-1] + 1 + rest, n_images))]
        k = int(f[-1]) + 1 + rest
    return out


def match_frames(step_times, image_times, tol=None) -> np.ndarray:
    """Image index of every trajectory step, matched by time: the nearest image within `tol` (default a quarter of the
    shortest image interval), or -1 where there is none (outside the image sequence, NaT).

    `step_times` has any shape and `image_times` is ascending; both datetime64 (`tol` a timedelta64) or both numbers.
    """
    t, s = np.asarray(image_times), np.asarray(step_times)
    if tol is None:
        tol = np.diff(t).min() / 4
    hi = np.clip(np.searchsorted(t, s), 0, t.size - 1)
    lo = np.clip(hi - 1, 0, t.size - 1)
    near = np.where(np.abs(s - t[lo]) < np.abs(t[hi] - s), lo, hi)
    return np.where(np.abs(t[near] - s) <= tol, near, -1)


def local_contrast(frame, dx: float, fwhm: float = 8.0, fraction: float = 0.5, log: bool = True, floor: float = 0.01) -> np.ndarray:
    """Cloud texture for display: the (log) image minus `fraction` x its Gaussian local mean of full width `fwhm` (same
    units as the pixel size `dx`). Removing part of the large-scale brightness, not all of it, keeps the eye and the
    eyewall apart while the small-scale texture shows. NaN-aware; needs scipy.
    """
    from scipy import ndimage

    f = np.log(np.clip(frame, floor, None)) if log else np.asarray(frame, float)
    if not fraction:
        return f
    ok = np.isfinite(f)
    sigma = fwhm / 2.355 / dx
    weight = ndimage.gaussian_filter(ok.astype(float), sigma)
    mean = ndimage.gaussian_filter(np.where(ok, f, 0.0), sigma) / np.maximum(weight, 1e-3)
    return np.where(ok, f - fraction * mean, np.nan)


def animate_trajectories(
    images, x, y, xloc, yloc, frame_index, *, times=None, valid=None, panels=None, stride=1, checker=False,
    color="yellow", cmap="plasma", clim=None, color_label=None, head_color=None, linewidth=2.0, tail_width=0.1, size=14.0,
    omega=0.0, center=None, ref_step=None, rings=(), image_transform=None, standardize=True, vlim=None,
    image_cmap="gray", image_label=None, rest=1, regular_tol=0.25, title="", axes=None,
) -> TrajectoryMovie:
    """Build the animation of tracked parcels moving along their trajectories over the images.

    Every trajectory set (one per AMV time) plays as a cycle (see `movie_schedule`): through the images its steps fall
    on, the parcels move and leave trails that thin smoothly with age; then `rest` images are shown alone and the next
    set begins. The movie runs at the image cadence and shows every image once.

    Parameters
    ----------
    images : (nt, ny, nx) array-like
        The image sequence. `images[k]` is read only when image k is drawn, so an xarray DataArray or a netCDF variable
        never has to fit in memory.
    x, y : (nx,), (ny,) arrays
        Pixel-center coordinates of the images (a regular grid, either order). Positions are in the same units; with km
        (and `times`) color="speed" is in m/s.
    xloc, yloc : (nset, nstep, ...) arrays
        Parcel positions: one set per AMV time (openTCAMV's `it`), its steps in time order, then any number of track
        dimensions (e.g. the seed grid (y, x)). NaN where not tracked.
    frame_index : (nset, nstep) int array
        Image index of every step (`match_frames`). A set with an index outside the images, or not increasing along the
        steps, is skipped.
    times : (nt,) datetime64 or seconds, optional
        Image times. They label the frames, set the trail ages, and are needed for `omega` and color="speed".
    valid : (nset, ...) bool array, optional
        Tracks to draw, e.g. `np.isfinite(vx)` (default: those with any finite position).
    panels : list of dict, optional
        One axes each. The window is `xlim`/`ylim`, or `center`/`half` for a square; optional `title`, `stride`,
        `checker`, `linewidth`, `size` and `vlim` override the arguments below. Default: one panel over the whole
        image. The first of several panels outlines the others.
    stride, checker
        Draw every `stride`-th track along each track dimension (int, or one per dimension); `checker` then keeps every
        other one of those in a checkerboard (spacing x sqrt 2).
    color
        A matplotlib color; "speed" for each trajectory's start-to-end speed in the drawn frame (x 1000: km and s ->
        m/s); or an (nset, ...) array of values. Values are colormapped with `cmap` over `clim` (default: 0 to the 98th
        percentile of the speeds, or the 2nd-98th percentiles of the values), labeled `color_label`.
    head_color
        Fill of the parcel markers, if not that of `color`: a matplotlib color, or "image" for the image's color under
        the parcel at the first step of its trajectory (where it first appears in the movie), in each panel's color
        scale. A parcel that follows its cloud then keeps the color of the cloud beneath it. The trails keep `color`.
    linewidth, tail_width, size
        Trail width at the parcel (points); it falls linearly with age to `tail_width` x `linewidth` at the age of the
        whole trajectory. `size` is the parcel marker area (points^2).
    omega, center, ref_step
        Draw each cycle in a frame rotating counterclockwise (cyclonically in the NH) at `omega` (rad/s) about `center`
        (default (0, 0)), coinciding with the image frame at step `ref_step` (default: the middle step); the images turn
        with it, and the colored speeds are relative to it. If `center` is given it is marked with a cross and `rings`
        (radii) are drawn about it.
    image_transform, standardize, vlim, image_cmap, image_label
        Image display: `image_transform(frame)` first (e.g. `local_contrast`); then, if `standardize`, the median over
        the first panel removed, the result divided by 1.2 x its 16-84 percentile spread and asinh-stretched. `vlim`
        gives the color limits of `image_cmap` (default +-2.4 when standardized, else the range of the first image over
        the first panel). Set `standardize=False` to show the values themselves (e.g. brightness temperature) and
        `image_label` to add a colorbar of them with that label: one under all panels, or one under each panel when a
        panel has its own `vlim`.
    rest : int
        Images shown alone after each cycle; they mark where the next set starts.
    regular_tol : float or None
        Skip a set whose image intervals depart from their median by more than this fraction: across a gap in the
        series the parcels would jump on screen. None keeps every set.
    title : str
        Prefix of the frame label (the figure's suptitle).
    axes : list of Axes, optional
        Draw into these (one per panel) instead of a new figure; the layout and any colorbar are then left to you.
    """
    nt = len(images)
    if nt == 0:
        raise ValueError("no images")
    x, y = np.asarray(x, float), np.asarray(y, float)
    xloc, yloc = np.asarray(xloc, float), np.asarray(yloc, float)
    nset, nstep = xloc.shape[:2]
    X, Y = xloc.reshape(nset, nstep, -1), yloc.reshape(nset, nstep, -1)
    fi = np.asarray(frame_index).astype(int).reshape(nset, nstep)
    ok = (np.isfinite(X) & np.isfinite(Y)).any(1) if valid is None else np.asarray(valid, bool).reshape(nset, -1)
    by_speed = isinstance(color, str) and color == "speed"
    image_heads = isinstance(head_color, str) and head_color == "image"
    values = None if isinstance(color, str) else np.asarray(color, float).reshape(nset, -1)
    mapped = by_speed or values is not None
    if times is None and (omega or by_speed):
        raise ValueError("`times` is needed for `omega` and for color='speed'")
    tt = np.arange(nt, dtype=float) if times is None else _seconds(times)
    ref = nstep // 2 if ref_step is None else int(ref_step)
    rc = (0.0, 0.0) if center is None else (float(center[0]), float(center[1]))
    nsub = int(np.ceil(16 / max(nstep - 1, 1)))                 # pieces per step: ~16 width levels along a trajectory

    usable = ((fi >= 0) & (fi < nt)).all(1) & (np.diff(fi, axis=1) > 0).all(1) & ok.any(1)
    if times is not None and regular_tol is not None:
        for j in np.flatnonzero(usable):
            dt = np.diff(tt[fi[j, 0]:fi[j, -1] + 1])
            usable[j] = dt.size == 0 or bool(np.all(np.abs(dt - np.median(dt)) <= regular_tol * np.median(dt)))
    if not usable.any():
        warnings.warn("no usable trajectory set (check frame_index, valid and the image times): images only")
    schedule = movie_schedule(fi, usable, nt, rest)

    def geometry(j):
        """Set j's positions in the drawn frame -- rotated into the frame co-rotating at `omega` that coincides with the
        image frame at the reference step -- with its step times and track values."""
        px, py, ts = X[j], Y[j], tt[fi[j]]
        if omega:
            th = -omega * (ts - ts[ref])[:, None]
            ux, uy = px - rc[0], py - rc[1]
            px, py = rc[0] + np.cos(th) * ux - np.sin(th) * uy, rc[1] + np.sin(th) * ux + np.cos(th) * uy
        val = values[j] if values is not None else _speed(px, py, ts) if by_speed else None
        return px, py, ts, val

    def trail(px, py, ts, s, q, lw):
        """Trails of tracks q up to step s, each step cut into `nsub` straight pieces whose width falls linearly with
        age: the pieces (n, 2, 2), their widths and their track (index into q), NaN pieces dropped."""
        a = np.linspace(0.0, 1.0, nsub + 1)[None, :, None]
        x0, y0 = px[:s, q][:, None], py[:s, q][:, None]
        xs_ = x0 + (px[1:s + 1, q][:, None] - x0) * a                         # (s, nsub + 1, nq)
        ys_ = y0 + (py[1:s + 1, q][:, None] - y0) * a
        seg = np.stack([np.stack([xs_[:, :-1], ys_[:, :-1]], -1), np.stack([xs_[:, 1:], ys_[:, 1:]], -1)], -2)
        tmid = ts[:s, None] + np.diff(ts[:s + 1])[:, None] * (np.arange(nsub) + 0.5) / nsub
        age = np.clip((ts[s] - tmid) / max(ts[-1] - ts[0], 1e-9), 0.0, 1.0)
        width = lw * (1.0 - (1.0 - tail_width) * age)                          # (s, nsub)
        seg = seg.transpose(2, 0, 1, 3, 4).reshape(-1, 2, 2)                  # track by track, tail to head
        width = np.broadcast_to(width, (len(q), s, nsub)).ravel()
        track = np.repeat(np.arange(len(q)), s * nsub)
        keep = np.isfinite(seg).all((1, 2))
        return seg[keep], width[keep], track[keep]

    cmap_ = plt.get_cmap(cmap)
    if mapped and clim is None:
        v = np.concatenate([geometry(j)[3][ok[j]] for j in np.flatnonzero(usable)] + [np.zeros(0)])
        v = v[np.isfinite(v)]
        lo, hi = np.percentile(v, [2, 98]) if v.size else (0.0, 1.0)
        clim = (0.0 if by_speed else lo, hi)
    norm = colors.Normalize(*clim) if mapped else None

    flip_x, flip_y = x[0] > x[-1], y[0] > y[-1]
    xs, ys = np.sort(x), np.sort(y)
    dx, dy = (xs[-1] - xs[0]) / (xs.size - 1), (ys[-1] - ys[0]) / (ys.size - 1)
    th_max = abs(omega) * max((abs(tt[k] - tt[fi[j, ref]]) for k, j, _ in schedule if j is not None), default=0.0)

    def crop(xlim, ylim, th):
        """Image rows and columns that can show in a window while the image turns by up to +-th about the center, and
        their extent."""
        ang = np.linspace(-th, th, 17)[:, None]
        cx = np.array([xlim[0], xlim[1], xlim[1], xlim[0]]) - rc[0]
        cy = np.array([ylim[0], ylim[0], ylim[1], ylim[1]]) - rc[1]
        bx, by = rc[0] + np.cos(ang) * cx - np.sin(ang) * cy, rc[1] + np.sin(ang) * cx + np.cos(ang) * cy
        ix = np.flatnonzero((xs >= bx.min() - dx) & (xs <= bx.max() + dx))
        iy = np.flatnonzero((ys >= by.min() - dy) & (ys <= by.max() + dy))
        if ix.size == 0 or iy.size == 0:                                      # window off the image
            ix, iy = np.arange(xs.size), np.arange(ys.size)
        extent = (xs[ix[0]] - dx / 2, xs[ix[-1]] + dx / 2, ys[iy[0]] - dy / 2, ys[iy[-1]] + dy / 2)
        return (slice(iy[0], iy[-1] + 1), slice(ix[0], ix[-1] + 1)), extent

    specs = []
    for p in panels or [{}]:
        if "center" in p:
            (cx, cy), h = p["center"], p["half"]
            xlim, ylim = (cx - h, cx + h), (cy - h, cy + h)
        else:
            xlim = p.get("xlim", (xs[0] - dx / 2, xs[-1] + dx / 2))
            ylim = p.get("ylim", (ys[0] - dy / 2, ys[-1] + dy / 2))
        specs.append(dict(xlim=tuple(xlim), ylim=tuple(ylim), title=p.get("title", ""), lw=p.get("linewidth", linewidth),
                          size=p.get("size", size), keep=_thin(xloc.shape[2:], p.get("stride", stride), p.get("checker", checker)),
                          vlim=p.get("vlim")))
    stats_crop = crop(specs[0]["xlim"], specs[0]["ylim"], 0.0)[0]
    per_panel = any(p["vlim"] is not None for p in specs)                    # image colorbar under each panel

    cache = {}

    def display(k):
        if k not in cache:
            f = np.ma.filled(np.ma.asarray(images[k], dtype=float), np.nan)
            if image_transform is not None:
                f = image_transform(f)
            f = f[::-1 if flip_y else 1, ::-1 if flip_x else 1]
            if standardize:
                r = f[stats_crop]
                r = r[np.isfinite(r)]
                if r.size:
                    p16, med, p84 = np.percentile(r, [16, 50, 84])
                    f = np.arcsinh((f - med) / (p84 - p16 if p84 > p16 else 1.0) / 1.2)
            if len(cache) >= 4:
                cache.pop(next(iter(cache)))
            cache[k] = f
        return cache[k]

    own = axes is None
    if own:
        ncol = 1 if len(specs) == 1 else 2
        nrow = -(-len(specs) // ncol)
        bars = 0 if image_label is None else nrow if per_panel else 1          # rows of image colorbars
        fig, grid = plt.subplots(nrow, ncol, figsize=(4.5 * ncol + 0.9 * mapped, 4.7 * nrow + 0.8 * bars), layout="constrained",
                                 squeeze=False)
        for extra in grid.ravel()[len(specs):]:
            extra.remove()
        axes = list(grid.ravel()[:len(specs)])
    else:
        axes = list(np.ravel(axes))
        fig = axes[0].figure

    shade = plt.get_cmap(image_cmap).copy()
    shade.set_bad("0.35")
    first = display(schedule[0][0])
    if vlim is None and standardize:
        vlim = (-2.4, 2.4)
    elif vlim is None:                                                        # one scale for every panel
        r = first[stats_crop]
        r = r[np.isfinite(r)]
        vlim = (float(r.min()), float(r.max())) if r.size else None
    for n, (ax, p) in enumerate(zip(axes, specs)):
        p["ax"] = ax
        p["crop"], extent = crop(p["xlim"], p["ylim"], th_max)
        lim = vlim if p["vlim"] is None else p["vlim"]
        p["im"] = ax.imshow(first[p["crop"]], origin="lower", extent=extent, cmap=shade, zorder=0,
                            vmin=None if lim is None else lim[0], vmax=None if lim is None else lim[1])
        p["lines"] = ax.add_collection(LineCollection([], capstyle="round", zorder=2), autolim=False)
        p["heads"] = ax.scatter(np.empty(0), np.empty(0), s=p["size"], facecolors="none" if mapped else color,
                                edgecolors="k", linewidths=0.6, zorder=3)
        if center is not None:
            ax.plot(*rc, "+", color="red", ms=10, mew=1.2, zorder=4)
            for r in rings:
                ax.add_patch(plt.Circle(rc, r, fill=False, ec="c", lw=0.6, ls="--", zorder=2))
        if n == 0:
            for o in specs[1:]:
                (x0, x1), (y0, y1) = o["xlim"], o["ylim"]
                ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="red", lw=1.2, zorder=4))
        ax.set(xlim=p["xlim"], ylim=p["ylim"], aspect="equal", xlabel="x (km)", ylabel="y (km)", facecolor="0.35")
        ax.tick_params(direction="in", top=True, right=True)
        ax.set_title((f"({chr(97 + n)}) " if len(specs) > 1 else "") + p["title"], loc="left")

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap_) if mapped else None
    if mappable is not None and own:
        default = ("Speed" + (" in the co-rotating frame" if omega else "") + " (m s$^{-1}$)") if by_speed else ""
        fig.colorbar(mappable, ax=axes, shrink=0.8, label=color_label if color_label is not None else default)
    if image_label is not None and own:
        targets = [(p["im"], p["ax"]) for p in specs] if per_panel else [(specs[0]["im"], axes)]
        for im, where in targets:
            fig.colorbar(im, ax=where, location="bottom", shrink=0.8 if per_panel else 0.6, aspect=25 if per_panel else 40,
                         extend="both", label=image_label)

    def frame_label(i):
        k, j, _ = schedule[i]
        parts = ([title] if title else []) + [_time_label(times, k)]
        if omega:
            mant, expo = f"{omega:.2e}".split("e")
            parts.append(rf"co-rotating at {float(mant):g}$\times$10$^{{{int(expo)}}}$ s$^{{-1}}$")
        if j is None:
            parts.append("image only between tracking cycles")                # about as long as the label below
        else:
            f0, f1 = fi[j, 0], fi[j, -1]
            parts.append(f"AMVs at {_time_label(times, fi[j, ref], short=True)}: tracking frame {k - f0 + 1}/{f1 - f0 + 1}")
        return " | ".join(parts)

    def start_colors(j):
        """For each panel, the RGBA of the displayed image under every track of set j at its first step (nearest pixel;
        in the image frame, which the drawn frame coincides with there)."""
        img = display(fi[j, 0])
        px, py = X[j, 0], Y[j, 0]
        inside = (np.abs(px - (xs[0] + xs[-1]) / 2) <= (xs[-1] - xs[0] + dx) / 2) & \
                 (np.abs(py - (ys[0] + ys[-1]) / 2) <= (ys[-1] - ys[0] + dy) / 2)
        ix = np.clip(np.rint((np.where(inside, px, xs[0]) - xs[0]) / dx).astype(int), 0, xs.size - 1)
        iy = np.clip(np.rint((np.where(inside, py, ys[0]) - ys[0]) / dy).astype(int), 0, ys.size - 1)
        value = np.where(inside, img[iy, ix], np.nan)
        return [p["im"].to_rgba(value) for p in specs]

    current = {}

    def draw(i):
        k, j, s = schedule[i]
        img = display(k)
        if j is not None and current.get("j") != j:
            current.update(j=j, geo=geometry(j), start=start_colors(j) if image_heads else None)
        th = -omega * (tt[k] - tt[fi[j, ref]]) if omega and j is not None else 0.0
        artists = []
        for n, p in enumerate(specs):
            p["im"].set_data(img[p["crop"]])
            if omega:
                p["im"].set_transform(transforms.Affine2D().rotate_around(rc[0], rc[1], th) + p["ax"].transData)
            heads, lines = p["heads"], p["lines"]
            heads.set_offsets(np.empty((0, 2)))
            lines.set_segments([])
            if j is not None:
                px, py, ts, val = current["geo"]
                (x0, x1), (y0, y1) = p["xlim"], p["ylim"]
                seen = ((px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)).any(0)
                q = np.flatnonzero(p["keep"] & ok[j] & seen)
                rgba = None if val is None else cmap_(norm(val[q]))
                xy = np.column_stack([px[s, q], py[s, q]])
                fin = np.isfinite(xy).all(1)
                heads.set_offsets(xy[fin])
                if image_heads:
                    heads.set_facecolor(current["start"][n][q][fin])
                else:
                    heads.set_facecolor(head_color or (color if rgba is None else rgba[fin]))
                if s >= 1:
                    seg, width, track = trail(px, py, ts, s, q, p["lw"])
                    if len(seg):
                        lines.set_segments(seg)
                        lines.set_linewidths(width)
                        lines.set_color(color if rgba is None else rgba[track])
            artists += [p["im"], lines, heads]
        label.set_text(frame_label(i))
        return artists + [label]

    label = fig.suptitle("")
    if own:                                       # freeze the layout on the longest label, shrunk to fit the figure if needed
        label.set_text(max((frame_label(i) for i in range(len(schedule))), key=len))
        fig.canvas.draw()
        width = label.get_window_extent().width / fig.bbox.width
        if width > 0.96:
            label.set_fontsize(label.get_fontsize() * 0.96 / width)
            fig.canvas.draw()
        fig.set_layout_engine("none")
        label.set(x=label.get_window_extent().x0 / fig.bbox.width, ha="left")   # every label starts there: no jumps
    draw(0)
    return TrajectoryMovie(fig, axes, schedule, draw, mappable)


def _thin(shape: tuple, stride, checker: bool) -> np.ndarray:
    """Flat mask over the track dimensions keeping every `stride`-th track per dimension, then (`checker`) every other
    one of those in a checkerboard."""
    steps = np.broadcast_to(np.asarray(stride, int), (len(shape),))
    idx = np.indices(shape)
    keep = np.ones(shape, bool)
    for i, st in zip(idx, steps):
        keep &= i % st == 0
    if checker:
        keep &= sum(i // st for i, st in zip(idx, steps)) % 2 == 0
    return keep.ravel()


def _speed(px: np.ndarray, py: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Start-to-end speed of every track (columns) over its finite steps, x 1000 (km and s -> m/s)."""
    fin = np.isfinite(px) & np.isfinite(py)
    i0, i1 = fin.argmax(0), len(ts) - 1 - fin[::-1].argmax(0)
    n = np.arange(px.shape[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.hypot(px[i1, n] - px[i0, n], py[i1, n] - py[i0, n]) * 1e3 / (ts[i1] - ts[i0])


def _seconds(times) -> np.ndarray:
    t = np.asarray(times)
    if np.issubdtype(t.dtype, np.datetime64):
        return (t - t[0]) / np.timedelta64(1, "s")
    return t.astype(float)


def _time_label(times, k: int, short: bool = False) -> str:
    if times is None:
        return f"image {k}"
    t = np.asarray(times)[k]
    if np.issubdtype(t.dtype, np.datetime64):
        text = np.datetime_as_string(t, unit="s")
        return text[11:] if short else text.replace("T", " ") + " UTC"
    return f"t = {t:g} s"
