import os
import sys
import urllib.request
import zipfile

import numpy as np
import onnx
import onnxruntime
from onnx import helper, numpy_helper
from ultralytics import YOLO
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
OSNET_HEAD_FILE = 'osnet_head.npz'
POOLING_OPERATION = 'GlobalAveragePool'
HEAD_OPERATION = 'Gemm'
BATCH_DIMENSION_NAME = 'batch'
PROJECTION_WEIGHT_NAME = 'cell_projection_weight'
PROJECTED_CELLS_NAME = 'projected_cells'
SPLIT_CHECK_BATCH_SIZE = 3
MAXIMUM_SPLIT_DIFFERENCE = 1e-4

# TensorRT detector engine (only on the machine that runs it, e.g. the Orin)
ENGINE_FLAG = '--engine'
ENGINE_IMAGE_SIZE = 640


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
    headNode = next(node for node in osnetModel.graph.node if node.op_type == HEAD_OPERATION)
    backboneModel = onnx.utils.Extractor(osnetModel).extract_model([osnetModel.graph.input[0].name],
                                                                   [poolingNode.input[0]])
    initializers = {initializer.name: numpy_helper.to_array(initializer) for initializer in osnetModel.graph.initializer}
    headWeight = initializers[headNode.input[1]]
    backboneModel.graph.initializer.append(
        numpy_helper.from_array(headWeight[:, :, None, None].astype(np.float32), PROJECTION_WEIGHT_NAME))
    backboneModel.graph.node.append(helper.make_node('Conv', [poolingNode.input[0], PROJECTION_WEIGHT_NAME],
                                                     [PROJECTED_CELLS_NAME]))
    projectedOutput = backboneModel.graph.output[0]
    projectedOutput.name = PROJECTED_CELLS_NAME
    for tensor in list(backboneModel.graph.input) + list(backboneModel.graph.output):
        tensor.type.tensor_type.shape.dim[0].dim_param = BATCH_DIMENSION_NAME
    del backboneModel.graph.value_info[:]
    onnx.save(backboneModel, os.path.join(MODELS_FOLDER, OSNET_BACKBONE_FILE))
    np.savez(os.path.join(MODELS_FOLDER, OSNET_HEAD_FILE), bias=initializers[headNode.input[2]])


def CheckOsnetSplit():
    providers = ['CPUExecutionProvider']
    osnetSession = onnxruntime.InferenceSession(os.path.join(MODELS_FOLDER, OSNET_MODEL_FILE), providers=providers)
    backboneSession = onnxruntime.InferenceSession(os.path.join(MODELS_FOLDER, OSNET_BACKBONE_FILE),
                                                   providers=providers)
    head = np.load(os.path.join(MODELS_FOLDER, OSNET_HEAD_FILE))
    testImages = np.random.default_rng(0).random((SPLIT_CHECK_BATCH_SIZE, 3, 256, 128), np.float32)
    wholeEmbeddings = np.concatenate([osnetSession.run(None, {osnetSession.get_inputs()[0].name: testImage[None]})[0]
                                      for testImage in testImages])
    featureMaps = backboneSession.run(None, {backboneSession.get_inputs()[0].name: testImages})[0]
    splitEmbeddings = np.maximum(featureMaps.mean(axis=(2, 3)) + head['bias'], 0.0)
    splitEmbeddings /= np.linalg.norm(splitEmbeddings, axis=1, keepdims=True)
    largestDifference = float(np.abs(wholeEmbeddings - splitEmbeddings).max())
    print(f'OSNet split: feature maps {list(featureMaps.shape)}, largest difference to the whole model '
          f'{largestDifference:.1e}')
    if largestDifference > MAXIMUM_SPLIT_DIFFERENCE:
        raise RuntimeError('the split OSNet does not give the same embedding as the whole model')


def ExportDetectorEngine():
    detectorModel = YOLO(os.path.join(MODELS_FOLDER, YOLO_MODEL_FILE))
    enginePath = detectorModel.export(format='engine', half=True, imgsz=ENGINE_IMAGE_SIZE, device=0)
    print(f'TensorRT detector: {enginePath} (set detectorModelFile to its name, then colcon build)')


def main():
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    FetchYoloModel()
    FetchOsnetModel()
    SplitOsnetModel()
    CheckOsnetSplit()
    if ENGINE_FLAG in sys.argv:
        ExportDetectorEngine()


main()
