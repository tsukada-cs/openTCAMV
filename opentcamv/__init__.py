"""openTCAMV core package: TC-specific AMV derivation on top of pyVTTrac v2.

Importing this package eagerly imports all public submodules, so a single
``import opentcamv`` is enough to use the fully-qualified form::

    import opentcamv

    ds = opentcamv.data.load(...)
    parser = opentcamv.cli.build_parser()

The exception is `trajectory_movie`, which needs the optional matplotlib:
import it explicitly (``from opentcamv.trajectory_movie import
animate_trajectories``).
"""

from . import (
    aggregate,
    cli,
    concat,
    data,
    finalize,
    output,
    params,
    records,
    rotation,
    screening,
    tracking,
)

__version__ = "2.0.0"

__all__ = [
    "aggregate",
    "cli",
    "concat",
    "data",
    "finalize",
    "output",
    "params",
    "records",
    "rotation",
    "screening",
    "tracking",
    "__version__",
]
