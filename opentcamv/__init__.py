"""openTCAMV core package: TC-specific AMV derivation on top of pyVTTrac v2.

Importing this package eagerly imports all public submodules, so a single
``import opentcamv`` is enough to use the fully-qualified form::

    import opentcamv

    ds = opentcamv.data.load(...)
    parser = opentcamv.cli.build_parser()
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
