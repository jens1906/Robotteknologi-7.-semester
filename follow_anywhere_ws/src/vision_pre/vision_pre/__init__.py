"""Vision package.

Scripts installed by a plain `colcon build` run with the system python, which has no timm or
onnxruntime: those only live in the workspace .venv. Its site-packages is appended at the END of
sys.path, so the system numpy / torch / OpenCV keep priority (the venv's numpy 2 and OpenCV 5
break cv_bridge) and only packages that are missing otherwise are taken from the venv.
"""
import glob
import os
import sys


def _append_workspace_venv():
    folder = os.path.dirname(os.path.abspath(__file__))
    while True:
        for site in glob.glob(os.path.join(folder, '.venv', 'lib', 'python*', 'site-packages')):
            if site not in sys.path:
                sys.path.append(site)
            return
        parent = os.path.dirname(folder)
        if parent == folder:
            return
        folder = parent


_append_workspace_venv()
