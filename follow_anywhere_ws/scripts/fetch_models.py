import os
import urllib.request
import zipfile

import numpy as np
import onnx
import onnxruntime
from ultralytics.utils.downloads import attempt_download_asset

# Models
MODELS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'src', 'perception', 'resource', 'models')
YOLO_MODEL_FILE = 'yolov8n-pose.pt'
OSNET_ARCHIVE_URL = ('https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/'
                     'osnet/releases/v0.64.0/osnet-onnx-float.zip')
OSNET_FILES = ('osnet.onnx', 'osnet.data')

# OSNet split into backbone + head
OSNET_MODEL_FILE = 'osnet.onnx'
OSNET_BACKBONE_FILE = 'osnet_backbone.onnx'
OSNET_HEAD_FILE = 'osnet_head.onnx'
POOLING_OPERATION = 'GlobalAveragePool'
MAXIMUM_SPLIT_DIFFERENCE = 1e-4


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


def SplitOsnetModel():
    osnetModel = onnx.load(os.path.join(MODELS_FOLDER, OSNET_MODEL_FILE))
    poolingNode = [node for node in osnetModel.graph.node if node.op_type == POOLING_OPERATION][-1]
    imageName = osnetModel.graph.input[0].name
    embeddingName = osnetModel.graph.output[0].name
    extractor = onnx.utils.Extractor(osnetModel)
    onnx.save(extractor.extract_model([imageName], [poolingNode.input[0]]),
              os.path.join(MODELS_FOLDER, OSNET_BACKBONE_FILE))
    onnx.save(extractor.extract_model([poolingNode.output[0]], [embeddingName]),
              os.path.join(MODELS_FOLDER, OSNET_HEAD_FILE))


def CheckOsnetSplit():
    providers = ['CPUExecutionProvider']
    osnetSession = onnxruntime.InferenceSession(os.path.join(MODELS_FOLDER, OSNET_MODEL_FILE), providers=providers)
    backboneSession = onnxruntime.InferenceSession(os.path.join(MODELS_FOLDER, OSNET_BACKBONE_FILE),
                                                   providers=providers)
    headSession = onnxruntime.InferenceSession(os.path.join(MODELS_FOLDER, OSNET_HEAD_FILE), providers=providers)
    testImage = np.random.default_rng(0).random((1, 3, 256, 128), np.float32)
    wholeEmbedding = osnetSession.run(None, {osnetSession.get_inputs()[0].name: testImage})[0]
    featureMap = backboneSession.run(None, {backboneSession.get_inputs()[0].name: testImage})[0]
    pooledFeatures = featureMap.mean(axis=(2, 3), keepdims=True)
    splitEmbedding = headSession.run(None, {headSession.get_inputs()[0].name: pooledFeatures})[0]
    largestDifference = float(np.abs(wholeEmbedding - splitEmbedding).max())
    print(f'OSNet split: feature map {list(featureMap.shape)}, largest difference to the whole model '
          f'{largestDifference:.1e}')
    if largestDifference > MAXIMUM_SPLIT_DIFFERENCE:
        raise RuntimeError('the split OSNet does not give the same embedding as the whole model')


def main():
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    FetchYoloModel()
    FetchOsnetModel()
    SplitOsnetModel()
    CheckOsnetSplit()


main()
