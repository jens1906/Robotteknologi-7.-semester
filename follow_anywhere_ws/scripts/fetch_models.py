import os
import urllib.request
import zipfile

from ultralytics.utils.downloads import attempt_download_asset

# Models
MODELS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'src', 'perception', 'resource', 'models')
YOLO_MODEL_FILE = 'yolov8n-pose.pt'
OSNET_ARCHIVE_URL = ('https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/'
                     'osnet/releases/v0.64.0/osnet-onnx-float.zip')
OSNET_FILES = ('osnet.onnx', 'osnet.data')


def FetchYoloModel():
    attempt_download_asset(os.path.join(MODELS_FOLDER, YOLO_MODEL_FILE))


def FetchOsnetModel():
    archivePath = os.path.join(MODELS_FOLDER, 'osnet_download.zip')
    urllib.request.urlretrieve(OSNET_ARCHIVE_URL, archivePath)
    with zipfile.ZipFile(archivePath) as archive:
        for memberName in archive.namelist():
            if os.path.basename(memberName) in OSNET_FILES:
                targetPath = os.path.join(MODELS_FOLDER, os.path.basename(memberName))
                with archive.open(memberName) as source, open(targetPath, 'wb') as target:
                    target.write(source.read())
    os.remove(archivePath)


def main():
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    FetchYoloModel()
    FetchOsnetModel()


main()
