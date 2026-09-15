"""Package containing task implementations for various robotic environments."""

import os
import toml

# Conveniences to other module directories via relative paths
LAB_TASKS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
"""Path to the extension source directory."""

LAB_TASKS_METADATA = toml.load(os.path.join(LAB_TASKS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = LAB_TASKS_METADATA["package"]["version"]

##
# Register Gym environments.
##

# This branch targets Wolf. Upstream prototype tasks include incomplete imports;
# register the supported Wolf environments explicitly instead of importing all demos.
from .manager_based.locomotion.velocity import wolf_env  # noqa: F401
