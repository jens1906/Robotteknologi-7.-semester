import importlib.util
import os
import sys
import tempfile
import urllib.request

import numpy as np
import onnx
import onnxruntime
import torch
from ament_index_python.packages import PackageNotFoundError, get_package_prefix
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset

# Models
MODELS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'src', 'perception', 'resource', 'models')
YOLO_MODEL_FILE = 'yolo26n-pose.pt'

# OSNet-AIN: torchreid (MIT) model code at a fixed commit + its MSMT17 weights, split into backbone + head
TORCHREID_SOURCE_URL = ('https://raw.githubusercontent.com/KaiyangZhou/deep-person-reid/'
                        'f8cd150fdf77e8d9e1ed143b7f308c2c609ded50/torchreid/models/{sourceFile}')
GOOGLE_DRIVE_URL = 'https://drive.google.com/uc?export=download&id={fileId}'
OSNET_AIN_SOURCE_FILE = 'osnet_ain.py'
OSNET_AIN_MODEL_NAME = 'osnet_ain_x1_0'
OSNET_AIN_WEIGHTS_FILE_ID = '1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal'
OSNET_BACKBONE_FILE = 'osnet_ain_msmt17_backbone.onnx'
OSNET_HEAD_FILE = 'osnet_ain_msmt17_head.npz'
OSNET_INPUT_SIZE = (256, 128)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BATCH_DIMENSION_NAME = 'batch'
ONNX_OPSET = 17
SPLIT_CHECK_BATCH_SIZE = 3
MAXIMUM_SPLIT_DIFFERENCE = 1e-4

# Image transport plugins (ROS apt packages, uv cannot install them): without them the bags record no camera images
IMAGE_TRANSPORT_PLUGINS = ('compressed_image_transport', 'compressed_depth_image_transport')
IMAGE_TRANSPORT_INSTALL_COMMAND = 'sudo apt install ros-jazzy-image-transport-plugins'

# TensorRT detector engine (only on the machine that runs it, e.g. the Orin)
ENGINE_FLAG = '--engine'
ENGINE_IMAGE_SIZE = 640


class SplitOsnetBackbone(torch.nn.Module):
    def __init__(self, reidModel, projectionWeight):
        super().__init__()
        self.reidModel = reidModel
        self.projection = torch.nn.Conv2d(projectionWeight.shape[1], projectionWeight.shape[0], 1, bias=False)
        self.projection.weight.data = projectionWeight[:, :, None, None]
        self.register_buffer('inputMean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('inputStd', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, image):
        return self.projection(self.reidModel.featuremaps((image - self.inputMean) / self.inputStd))


def CheckImageTransportPlugins():
    missingPlugins = []
    for pluginPackage in IMAGE_TRANSPORT_PLUGINS:
        try:
            get_package_prefix(pluginPackage)
        except PackageNotFoundError:
            missingPlugins.append(pluginPackage)
    if missingPlugins:
        sys.exit(f'Missing {", ".join(missingPlugins)}: the bags would record no camera images. '
                 f'Install with `{IMAGE_TRANSPORT_INSTALL_COMMAND}` (ROS sourced), then run again.')


def FetchYoloModel():
    attempt_download_asset(os.path.join(MODELS_FOLDER, YOLO_MODEL_FILE))


def LoadOsnetAinModel(downloadFolder):
    sourcePath = os.path.join(downloadFolder, OSNET_AIN_SOURCE_FILE)
    urllib.request.urlretrieve(TORCHREID_SOURCE_URL.format(sourceFile=OSNET_AIN_SOURCE_FILE), sourcePath)
    moduleSpec = importlib.util.spec_from_file_location('torchreid_osnet_ain', sourcePath)
    torchreidModule = importlib.util.module_from_spec(moduleSpec)
    moduleSpec.loader.exec_module(torchreidModule)
    weightsPath = os.path.join(downloadFolder, f'{OSNET_AIN_MODEL_NAME}.pth')
    urllib.request.urlretrieve(GOOGLE_DRIVE_URL.format(fileId=OSNET_AIN_WEIGHTS_FILE_ID), weightsPath)
    stateDict = torch.load(weightsPath, map_location='cpu', weights_only=True)
    stateDict = {name.removeprefix('module.'): weights for name, weights in stateDict.items()}
    classCount = stateDict['classifier.weight'].shape[0]
    reidModel = getattr(torchreidModule, OSNET_AIN_MODEL_NAME)(num_classes=classCount, pretrained=False)
    reidModel.load_state_dict(stateDict)
    return reidModel.eval()


def ExportOsnetAinModel():
    with tempfile.TemporaryDirectory() as downloadFolder:
        reidModel = LoadOsnetAinModel(downloadFolder)
    fusedLinear = torch.nn.utils.fusion.fuse_linear_bn_eval(reidModel.fc[0], reidModel.fc[1])
    splitBackbone = SplitOsnetBackbone(reidModel, fusedLinear.weight.detach()).eval()
    headBias = fusedLinear.bias.detach().numpy().astype(np.float32)
    testImages = torch.rand(SPLIT_CHECK_BATCH_SIZE, 3, *OSNET_INPUT_SIZE, generator=torch.Generator().manual_seed(0))
    backbonePath = os.path.join(MODELS_FOLDER, OSNET_BACKBONE_FILE)
    torch.onnx.export(splitBackbone, testImages, backbonePath, input_names=['image'], output_names=['cells'],
                      dynamic_axes={'image': {0: BATCH_DIMENSION_NAME}, 'cells': {0: BATCH_DIMENSION_NAME}},
                      opset_version=ONNX_OPSET, dynamo=False)
    with torch.no_grad():
        cellShape = splitBackbone(testImages).shape[1:]
        wholeEmbeddings = reidModel((testImages - splitBackbone.inputMean) / splitBackbone.inputStd).numpy()
    backboneModel = onnx.load(backbonePath)
    for dimension, size in zip(backboneModel.graph.output[0].type.tensor_type.shape.dim[1:], cellShape):
        dimension.dim_value = size
    onnx.save(backboneModel, backbonePath)
    np.savez(os.path.join(MODELS_FOLDER, OSNET_HEAD_FILE), bias=headBias)
    CheckOsnetSplit(backbonePath, headBias, testImages.numpy(), wholeEmbeddings)


def CheckOsnetSplit(backbonePath, headBias, testImages, wholeEmbeddings):
    backboneSession = onnxruntime.InferenceSession(backbonePath, providers=['CPUExecutionProvider'])
    featureMaps = backboneSession.run(None, {'image': testImages})[0]
    splitEmbeddings = np.maximum(featureMaps.mean(axis=(2, 3)) + headBias, 0.0)
    splitEmbeddings /= np.linalg.norm(splitEmbeddings, axis=1, keepdims=True)
    wholeEmbeddings = wholeEmbeddings / np.linalg.norm(wholeEmbeddings, axis=1, keepdims=True)
    largestDifference = float(np.abs(wholeEmbeddings - splitEmbeddings).max())
    print(f'OSNet-AIN split: feature maps {list(featureMaps.shape)}, largest difference to the whole model '
          f'{largestDifference:.1e}')
    if largestDifference > MAXIMUM_SPLIT_DIFFERENCE:
        raise RuntimeError('the split OSNet-AIN does not give the same embedding as the whole model')


def ExportDetectorEngine():
    detectorModel = YOLO(os.path.join(MODELS_FOLDER, YOLO_MODEL_FILE))
    enginePath = detectorModel.export(format='engine', half=True, imgsz=ENGINE_IMAGE_SIZE, device=0)
    print(f'TensorRT detector: {enginePath} (set detectorModelFile to its name, then colcon build)')


def main():
    CheckImageTransportPlugins()
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    FetchYoloModel()
    ExportOsnetAinModel()
    if ENGINE_FLAG in sys.argv:
        ExportDetectorEngine()


main()
