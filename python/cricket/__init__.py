from __future__ import annotations

from pathlib import Path

from . import _core_ext as _ext
from ._core_ext import RobotInfo, GenOptions, GenResult, generate_robot_source


_INSTALL_DIR = Path(_ext.__file__).resolve().parent


def resources_dir() -> Path:
    """Filesystem path to cricket's installed `resources/` tree.

    Holds the inja templates (`templates/fk_template.hh`, `templates/ccfk_template.hh`)
    and the bundled robot URDF/SRDF samples.
    """
    return _INSTALL_DIR / "resources"


def cmake_dir() -> Path:
    """Filesystem path to cricket's installed CMake package config.

    Add this to `CMAKE_PREFIX_PATH` for downstream `find_package(cricket)` to work.
    """
    return _INSTALL_DIR / "cmake"


__all__ = [
    "RobotInfo",
    "GenOptions",
    "GenResult",
    "generate_robot_source",
    "resources_dir",
    "cmake_dir",
]
