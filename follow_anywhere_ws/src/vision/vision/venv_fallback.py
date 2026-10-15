"""Find packages that only live in the workspace's .venv when a node runs with another python.

The installed ROS scripts run with whichever python built them. A plain `colcon build` uses
/usr/bin/python3, which has no onnxruntime / motmetrics (only in follow_anywhere_ws/.venv),
and activating the venv does not change that. So when an import fails, look for the package in a
.venv above this file and add its site-packages. It goes at the end of sys.path, so the system
packages (numpy, torch, opencv ...) keep priority.
"""
import glob
import importlib
import os
import sys


def ensure(module):
    """Import and return `module`, falling back to the workspace .venv if it is missing."""
    try:
        return importlib.import_module(module)
    except ImportError as first:
        top = module.split('.')[0]
        here = os.path.dirname(os.path.abspath(__file__))
        while True:
            for site in glob.glob(os.path.join(here, '.venv', 'lib', 'python*', 'site-packages')):
                if os.path.isdir(os.path.join(site, top)) and site not in sys.path:
                    sys.path.append(site)
                    return importlib.import_module(module)
            parent = os.path.dirname(here)
            if parent == here:
                raise first
            here = parent
