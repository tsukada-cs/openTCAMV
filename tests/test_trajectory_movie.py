"""`opentcamv.trajectory_movie`: the self-contained trajectory animation, and its CLI `31_animate_trajectories.py`.

The geometry checks use analytic motion: parcels in rigid rotation must stand still in a frame co-rotating at the
same rate (both the parcels and the image), and a uniform translation must color every parcel by its speed.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from matplotlib.collections import LineCollection, PathCollection
from PIL import Image

from conftest import SCRIPTS_DIR, make_uniform_advection_frames, run_tracking
from opentcamv.trajectory_movie import animate_trajectories, match_frames, movie_schedule

MODULE = Path(__file__).resolve().parent.parent / "opentcamv" / "trajectory_movie.py"
T0 = np.datetime64("2020-01-01T00:00:00")
X = Y = np.linspace(-20.0, 20.0, 41)
GRID = np.linspace(-16.0, 16.0, 9)


def bothward_index(nt: int, ntrac: int) -> np.ndarray:
    """openTCAMV's bothward layout: a set centered on every image that has ntrac images on both sides."""
    centers = np.arange(ntrac, nt - ntrac)
    return centers[:, None] + np.arange(-ntrac, ntrac + 1)[None, :]


def moving_parcels(nt: int, ntrac: int, dt: float, omega: float = 0.0, u: float = 0.0, v: float = 0.0):
    """Parcels seeded on a 9 x 9 grid at each set's center time, in rigid rotation at `omega` (rad/s, CCW) plus a
    uniform translation (u, v) in m/s."""
    fi = bothward_index(nt, ntrac)
    tsec = np.arange(nt) * dt
    rel = (tsec[fi] - tsec[fi[:, ntrac]][:, None])[..., None, None]           # (nset, nstep, 1, 1)
    gx, gy = np.meshgrid(GRID, GRID)
    th = omega * rel
    xloc = np.cos(th) * gx - np.sin(th) * gy + u * rel / 1e3
    yloc = np.sin(th) * gx + np.cos(th) * gy + v * rel / 1e3
    times = T0 + (tsec * 1e3).astype("timedelta64[ms]")
    return times, fi, xloc, yloc


def artists(ax):
    heads = next(c for c in ax.collections if isinstance(c, PathCollection))
    lines = next(c for c in ax.collections if isinstance(c, LineCollection))
    return heads, lines


def test_module_is_self_contained(tmp_path):
    """The file must work when copied alone into a project: importing it pulls in nothing beyond numpy and matplotlib."""
    shutil.copy(MODULE, tmp_path / "trajectory_movie.py")
    code = ("import sys, trajectory_movie\n"
            "print([m for m in ('opentcamv', 'xarray', 'pandas', 'scipy', 'pyvttrac') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


def test_schedule_shows_every_image_once():
    fi = bothward_index(10, 1)                                    # sets centered on images 1..8
    sched = movie_schedule(fi, np.ones(len(fi), bool), 10, rest=1)
    assert [k for k, _, _ in sched] == list(range(10))
    assert sched[:5] == [(0, 0, 0), (1, 0, 1), (2, 0, 2), (3, None, None), (4, 4, 0)]


def test_schedule_skips_unusable_sets_and_holds_between_sparse_steps():
    fi = bothward_index(10, 1)
    usable = np.ones(len(fi), bool)
    usable[4] = False                                             # the set starting on image 4
    sched = movie_schedule(fi, usable, 10, rest=1)
    assert sched[4] == (4, None, None) and sched[5] == (5, 5, 0)
    # --traj_int 2: steps on every other image; between them the latest step is shown
    assert movie_schedule(np.array([[0, 2, 4]]), np.array([True]), 6, rest=1) == [
        (0, 0, 0), (1, 0, 0), (2, 0, 1), (3, 0, 1), (4, 0, 2), (5, None, None)]


def test_irregular_image_spacing_skips_the_set():
    secs = np.array([0, 60, 600, 660, 720, 780, 840], float)      # gap between images 1 and 2
    fi = bothward_index(7, 1)                                      # centers 1..5
    xloc = np.zeros((len(fi), 3, 1))
    movie = animate_trajectories(np.zeros((7, 41, 41)), X, Y, xloc, xloc, fi, times=secs)
    assert movie.schedule == [(0, None, None), (1, None, None), (2, 2, 0), (3, 2, 1), (4, 2, 2), (5, None, None), (6, None, None)]
    movie = animate_trajectories(np.zeros((7, 41, 41)), X, Y, xloc, xloc, fi, times=secs, regular_tol=None)
    assert movie.schedule[0] == (0, 0, 0)
    plt.close("all")


def test_match_frames():
    images = T0 + np.arange(5) * np.timedelta64(120, "s")
    steps = np.array([images[0] + np.timedelta64(3, "s"), images[3], images[1] + np.timedelta64(50, "s"),
                      np.datetime64("NaT"), images[4] + np.timedelta64(120, "s")])
    np.testing.assert_array_equal(match_frames(steps, images), [0, 3, -1, -1, -1])


def test_corotation_holds_rigidly_rotating_parcels_and_image_still():
    omega = 2e-3
    times, fi, xloc, yloc = moving_parcels(nt=9, ntrac=3, dt=60.0, omega=omega)
    movie = animate_trajectories(np.zeros((9, 41, 41)), X, Y, xloc, yloc, fi, times=times, omega=omega, center=(0, 0))
    ax = movie.axes[0]
    heads, _ = artists(ax)
    seeds = np.column_stack([a.ravel() for a in np.meshgrid(GRID, GRID)])
    p_ref = np.array([10.0, 5.0])
    checked = 0
    for i, (k, j, s) in enumerate(movie.schedule):
        movie.draw(i)
        if j is None:
            continue
        np.testing.assert_allclose(heads.get_offsets(), seeds, atol=1e-9)
        a = omega * (times[k] - times[fi[j, 3]]) / np.timedelta64(1, "s")  # image feature at p_ref at the AMV time
        p_k = [np.cos(a) * p_ref[0] - np.sin(a) * p_ref[1], np.sin(a) * p_ref[0] + np.cos(a) * p_ref[1]]
        np.testing.assert_allclose(ax.images[0].get_transform().transform(p_k), ax.transData.transform(p_ref), atol=1e-6)
        checked += 1
    assert checked == 7
    plt.close("all")


def test_speed_color_and_smooth_taper():
    times, fi, xloc, yloc = moving_parcels(nt=9, ntrac=3, dt=100.0, u=6.0, v=8.0)   # 10 m/s
    lw, tail = 2.0, 0.1
    movie = animate_trajectories(np.zeros((9, 41, 41)), X, Y, xloc, yloc, fi, times=times, color="speed",
                                 cmap="viridis", clim=(0, 30), linewidth=lw, tail_width=tail)
    heads, lines = artists(movie.axes[0])
    last = next(i for i, (_, j, s) in enumerate(movie.schedule) if j is not None and s == 6)
    movie.draw(last)
    np.testing.assert_allclose(heads.get_facecolors(), np.broadcast_to(plt.get_cmap("viridis")(1 / 3), heads.get_facecolors().shape))  # mid-bin of the 256-level LUT
    width = np.asarray(lines.get_linewidths()).reshape(len(heads.get_offsets()), -1)  # track by track, tail to head
    assert width.shape[1] >= 16                                    # many width levels, not a step function
    assert np.all(np.diff(width, axis=1) > 0)
    assert tail * lw < width.min() < 0.15 * lw and 0.95 * lw < width.max() <= lw
    plt.close("all")


def test_panels_select_tracks_by_window_stride_and_validity():
    times, fi, xloc, yloc = moving_parcels(nt=5, ntrac=1, dt=60.0, u=20.0)
    valid = np.ones((len(fi), 9, 9), bool)
    valid[:, 0, :] = False                                         # bottom seed row rejected
    panels = [dict(stride=2), dict(checker=True), dict(center=(10, 10), half=3)]
    movie = animate_trajectories(np.zeros((5, 41, 41)), X, Y, xloc, yloc, fi, times=times, valid=valid, panels=panels)
    movie.draw(1)
    _, j, _ = movie.schedule[1]
    counts = [len(artists(ax)[0].get_offsets()) for ax in movie.axes]
    in_zoom = ((np.abs(xloc[j] - 10) <= 3) & (np.abs(yloc[j] - 10) <= 3)).any(0) & valid[j]
    assert counts == [20, 36, in_zoom.sum()]                       # 5 x 5 minus the bottom row; checkerboard of 8 rows
    assert 0 < counts[2] < 81
    plt.close("all")


def test_user_axes_and_gif(tmp_path):
    times, fi, xloc, yloc = moving_parcels(nt=6, ntrac=1, dt=60.0, u=20.0)
    images = np.random.default_rng(0).random((6, 41, 41))
    fig, ax = plt.subplots()
    movie = animate_trajectories(images, X, Y, xloc, yloc, fi, times=times, axes=[ax])
    assert movie.fig is fig and movie.axes == [ax]
    movie.save(tmp_path / "m.gif", fps=4, dpi=40)
    assert Image.open(tmp_path / "m.gif").n_frames == len(movie.schedule) == 6
    plt.close("all")


@pytest.mark.parametrize("seeds", [["--xgran=-6:6", "--ygran=-6:6", "--xint", "2", "--yint", "2"],
                                   ["--polar", "--rgran", "2:6", "--rint", "2", "--nath", "12"]], ids=["cartesian", "polar"])
def test_cli_end_to_end(tmp_path, seeds):
    """10_ on a uniformly advected field, then 31_ on its output; the image file comes from the `ifn` attribute."""
    frames = make_uniform_advection_frames(6, 81, 81, 0.25, 0.25, -10.0, -10.0, 150.0, 10.0, 5.0)
    ofn = run_tracking(tmp_path, frames, "Z", ["--ns", "9", "--ntrac", "1", "--Sth0", "0.5", "--Sth1", "0.5", "--Cth", "0",
                                                "--Vs", "25", "--Vd", "0", "--vlim", "0", "--traj_int", "1", "--revrot", "0.0", *seeds])
    movie = tmp_path / "movie.gif"
    result = subprocess.run([sys.executable, str(SCRIPTS_DIR / "31_animate_trajectories.py"), str(ofn), "-o", str(movie),
                             "--stride", "1", "--color", "speed", "--zoom", "0", "0", "4", "--dpi", "40"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert Image.open(movie).n_frames == xr.open_dataset(tmp_path / "in.nc").sizes["t"]
