import glob
import os
import sys

# Workspace virtual environment
VENV_FOLDER_NAME = '.venv'


def AppendWorkspaceVenvToPath():
    currentFolder = os.path.dirname(os.path.realpath(__file__))
    while currentFolder != os.path.dirname(currentFolder):
        sitePackagesFolders = glob.glob(os.path.join(currentFolder, VENV_FOLDER_NAME, 'lib', 'python*', 'site-packages'))
        if sitePackagesFolders:
            sys.path.append(sitePackagesFolders[0])
            return
        currentFolder = os.path.dirname(currentFolder)


os.environ.setdefault('YOLO_AUTOINSTALL', 'False')
AppendWorkspaceVenvToPath()
