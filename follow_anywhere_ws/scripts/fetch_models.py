import importlib.util
import os
import sys
import tempfile
import urllib.request
import zipfile

import numpy as np
import onnx
import onnxruntime
import torch
from ament_index_python.packages import PackageNotFoundError, get_package_prefix
from onnx import helper, numpy_helper
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset

# Models
MODELS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'src', 'perception', 'resource', 'models')
YOLO_MODEL_FILE = 'yolo26n-pose.pt'
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

# ReID candidates (perception re-evaluation, only with --reid-candidates). torchreid (MIT) model code at a fixed
# commit + its MSMT17 combineall weights; DINOv2 ViT-S/14 (Apache-2.0) at a fixed commit
REID_CANDIDATES_FLAG = '--reid-candidates'
TORCHREID_SOURCE_URL = ('https://raw.githubusercontent.com/KaiyangZhou/deep-person-reid/'
                        'f8cd150fdf77e8d9e1ed143b7f308c2c609ded50/torchreid/models/{sourceFile}')
GOOGLE_DRIVE_URL = 'https://drive.google.com/uc?export=download&id={fileId}'
TORCHREID_CANDIDATES = {
    'osnet_msmt17': ('osnet.py', 'osnet_x1_0', '1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x'),
    'osnet_ibn_msmt17': ('osnet.py', 'osnet_ibn_x1_0', '1q3Sj2ii34NlfxA4LvmHdWO_75NDRmECJ'),
    'osnet_ain_msmt17': ('osnet_ain.py', 'osnet_ain_x1_0', '1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal'),
}
DINOV2_HUB_REPOSITORY = 'facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8'
DINOV2_CANDIDATE = 'dinov2_vits14'
DINOV2_INPUT_SIZE = (224, 112)
DINOV2_PATCH_SIZE = 14
OSNET_CANDIDATE_INPUT_SIZE = (256, 128)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BACKBONE_FILE_ENDING = '_backbone.onnx'
HEAD_FILE_ENDING = '_head.npz'
ONNX_OPSET = 17

# Image transport plugins (ROS apt packages, uv cannot install them): without them the bags record no camera images
IMAGE_TRANSPORT_PLUGINS = ('compressed_image_transport', 'compressed_depth_image_transport')
IMAGE_TRANSPORT_INSTALL_COMMAND = 'sudo apt install ros-jazzy-image-transport-plugins'

# TensorRT detector engine (only on the machine that runs it, e.g. the Orin)
ENGINE_FLAG = '--engine'
ENGINE_IMAGE_SIZE = 640


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


class SplitTorchreidBackbone(torch.nn.Module):
    def __init__(self, reidModel, projectionWeight):
        super().__init__()
        self.reidModel = reidModel
        self.projection = torch.nn.Conv2d(projectionWeight.shape[1], projectionWeight.shape[0], 1, bias=False)
        self.projection.weight.data = projectionWeight[:, :, None, None]
        self.register_buffer('inputMean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('inputStd', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, image):
        return self.projection(self.reidModel.featuremaps((image - self.inputMean) / self.inputStd))


class SplitDinov2Backbone(torch.nn.Module):
    def __init__(self, dinoModel):
        super().__init__()
        self.dinoModel = dinoModel
        self.register_buffer('inputMean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('inputStd', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, image):
        patchTokens = self.dinoModel.forward_features((image - self.inputMean) / self.inputStd)['x_norm_patchtokens']
        cellRows, cellColumns = (side // DINOV2_PATCH_SIZE for side in DINOV2_INPUT_SIZE)
        return patchTokens.reshape(image.shape[0], cellRows, cellColumns, -1).permute(0, 3, 1, 2)


def LoadTorchreidModel(sourceFile, modelName, weightsFileId, downloadFolder):
    sourcePath = os.path.join(downloadFolder, sourceFile)
    urllib.request.urlretrieve(TORCHREID_SOURCE_URL.format(sourceFile=sourceFile), sourcePath)
    moduleSpec = importlib.util.spec_from_file_location(f'torchreid_{os.path.splitext(sourceFile)[0]}', sourcePath)
    torchreidModule = importlib.util.module_from_spec(moduleSpec)
    moduleSpec.loader.exec_module(torchreidModule)
    weightsPath = os.path.join(downloadFolder, f'{modelName}.pth')
    urllib.request.urlretrieve(GOOGLE_DRIVE_URL.format(fileId=weightsFileId), weightsPath)
    stateDict = torch.load(weightsPath, map_location='cpu', weights_only=True)
    stateDict = {name.removeprefix('module.'): weights for name, weights in stateDict.items()}
    reidModel = getattr(torchreidModule, modelName)(num_classes=stateDict['classifier.weight'].shape[0],
                                                     pretrained=False)
    reidModel.load_state_dict(stateDict)
    return reidModel.eval()


def ExportTorchreidCandidate(candidateName, sourceFile, modelName, weightsFileId):
    with tempfile.TemporaryDirectory() as downloadFolder:
        reidModel = LoadTorchreidModel(sourceFile, modelName, weightsFileId, downloadFolder)
    fusedLinear = torch.nn.utils.fusion.fuse_linear_bn_eval(reidModel.fc[0], reidModel.fc[1])
    splitBackbone = SplitTorchreidBackbone(reidModel, fusedLinear.weight.detach()).eval()
    testImages = torch.rand(3, 3, *OSNET_CANDIDATE_INPUT_SIZE, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        wholeEmbeddings = reidModel((testImages - splitBackbone.inputMean) / splitBackbone.inputStd).numpy()
    ExportSplitBackbone(candidateName, splitBackbone, fusedLinear.bias.detach().numpy(), True, testImages,
                        wholeEmbeddings)


def ExportDinov2Candidate():
    dinoModel = torch.hub.load(DINOV2_HUB_REPOSITORY, DINOV2_CANDIDATE).eval()
    splitBackbone = SplitDinov2Backbone(dinoModel).eval()
    testImages = torch.rand(3, 3, *DINOV2_INPUT_SIZE, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        wholeEmbeddings = splitBackbone(testImages).mean(dim=(2, 3)).numpy()
    ExportSplitBackbone(DINOV2_CANDIDATE, splitBackbone, np.zeros(wholeEmbeddings.shape[1], np.float32), False,
                        testImages, wholeEmbeddings)


def ExportSplitBackbone(candidateName, splitBackbone, headBias, headRelu, testImages, wholeEmbeddings):
    backbonePath = os.path.join(MODELS_FOLDER, candidateName + BACKBONE_FILE_ENDING)
    torch.onnx.export(splitBackbone, testImages, backbonePath, input_names=['image'], output_names=['cells'],
                      dynamic_axes={'image': {0: BATCH_DIMENSION_NAME}, 'cells': {0: BATCH_DIMENSION_NAME}},
                      opset_version=ONNX_OPSET, dynamo=False)
    with torch.no_grad():
        cellShape = splitBackbone(testImages).shape[1:]
    backboneModel = onnx.load(backbonePath)
    for dimension, size in zip(backboneModel.graph.output[0].type.tensor_type.shape.dim[1:], cellShape):
        dimension.dim_value = size
    onnx.save(backboneModel, backbonePath)
    np.savez(os.path.join(MODELS_FOLDER, candidateName + HEAD_FILE_ENDING), bias=headBias.astype(np.float32),
             relu=headRelu)
    backboneSession = onnxruntime.InferenceSession(backbonePath, providers=['CPUExecutionProvider'])
    featureMaps = backboneSession.run(None, {'image': testImages.numpy()})[0]
    splitEmbeddings = featureMaps.mean(axis=(2, 3)) + headBias
    splitEmbeddings = np.maximum(splitEmbeddings, 0.0) if headRelu else splitEmbeddings
    splitEmbeddings /= np.linalg.norm(splitEmbeddings, axis=1, keepdims=True)
    wholeEmbeddings = wholeEmbeddings / np.linalg.norm(wholeEmbeddings, axis=1, keepdims=True)
    largestDifference = float(np.abs(wholeEmbeddings - splitEmbeddings).max())
    print(f'{candidateName} split: feature maps {list(featureMaps.shape)}, largest difference to the whole model '
          f'{largestDifference:.1e}')
    if largestDifference > MAXIMUM_SPLIT_DIFFERENCE:
        raise RuntimeError(f'the split {candidateName} does not give the same embedding as the whole model')


def ExportDetectorEngine():
    detectorModel = YOLO(os.path.join(MODELS_FOLDER, YOLO_MODEL_FILE))
    enginePath = detectorModel.export(format='engine', half=True, imgsz=ENGINE_IMAGE_SIZE, device=0)
    print(f'TensorRT detector: {enginePath} (set detectorModelFile to its name, then colcon build)')


def main():
    CheckImageTransportPlugins()
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    FetchYoloModel()
    FetchOsnetModel()
    SplitOsnetModel()
    CheckOsnetSplit()
    if REID_CANDIDATES_FLAG in sys.argv:
        for candidateName, (sourceFile, modelName, weightsFileId) in TORCHREID_CANDIDATES.items():
            ExportTorchreidCandidate(candidateName, sourceFile, modelName, weightsFileId)
        ExportDinov2Candidate()
    if ENGINE_FLAG in sys.argv:
        ExportDetectorEngine()


main()
