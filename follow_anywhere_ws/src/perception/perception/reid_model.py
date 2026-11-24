"""Describes how a person looks (re-identification). Plain python, no ROS. OSNet-AIN (MSMT17).

To include:
- Input: the colour image, the aligned depth image and the people found in it. Each person is cut out by their box,
  squeezed to the backbone's input size (256x128) and run through the backbone (all people of a frame in one batch).
  The result is a grid of 16x8 cells, each cell 512 numbers describing that patch of the person (top rows head,
  bottom rows feet). Input size, grid and feature size are read from the ONNX file.
- Visible cells (0..1 per cell): which cells really show this person.
  - Depth mask (reidDepthMask 'cells'): the share of the cell's pixels within depthMaskBand of the person's torso
    depth. Somebody standing in front (or the wall behind) is at another depth and drops out. Invalid depth (0)
    counts as visible. No depth image or no torso depth: every cell visible.
  - Keypoints: rows more than one cell below the lowest visible YOLO-pose keypoint are not visible (legs hidden).
- The OSNet head is Linear + BatchNorm + ReLU. scripts/fetch_models.py fused Linear + BatchNorm and put it into the
  backbone as a 1x1 convolution on every cell (a weighted average of the cells and the Linear part may be swapped,
  and so it runs on the GPU with the backbone). A comparison only averages cells, adds the bias (<model>_head.npz
  next to <model>_backbone.onnx), applies ReLU and normalises.
- Comparing two people only over the cells visible in both (shared area): average those cells in both, finish the
  head and L2-normalise; the distance between the two embeddings is 0 (same) to 2 (opposite). A full-body view of the leader can so be matched with a half-hidden
  leader (upper body against upper body). Shared area below minimumSharedArea: distance NaN (unknown, no evidence).
  With every cell visible this is exactly the whole OSNet's embedding.
- OSNet: ONNX backbone run with onnxruntime, loaded from the package's resource/models folder (made by
  scripts/fetch_models.py). TensorRT provider (FP32) when onnxruntime has it, else CUDA, else CPU with a warning.
  TensorRT builds an engine for this GPU on the first start (~40 s on the Orin) and keeps it in
  TENSORRT_CACHE_FOLDER, so later starts load it. FP32 because the thresholds were measured with FP32 distances
  (measured: TensorRT FP32 differs at most 0.0018, FP16 0.0026). A batch of 4: CUDA 24 ms, TensorRT FP32 11 ms.
  On the GPU the crops always go through in batches of GPU_BATCH_SIZE, padded with empty images: the CUDA provider
  re-tunes cuDNN whenever the batch size changes (measured ~6 s per change of the number of people) and a TensorRT
  engine is built for one batch size. One empty batch runs at the start, so the engine build and the warm-up
  happen before the first person is compared.
- Only computes what it is asked for: the decision "is this the leader" is in leader_manager.
"""

import os
from dataclasses import dataclass
from types import SimpleNamespace

import cv2
import numpy as np
import onnxruntime

# Parameter defaults (set in config/perception_params.yaml)
REQUESTED_DEVICE = 'auto'
REID_THREADS = 2
REID_SETTING_DEFAULTS = {
    'reidDepthMask': 'cells',
    'depthMaskBand': 0.5,
}

# ReID backbone
BACKBONE_FILE_ENDING = '_backbone.onnx'
HEAD_FILE_ENDING = '_head.npz'
TENSORRT_PROVIDER = 'TensorrtExecutionProvider'
TENSORRT_CACHE_FOLDER = os.path.expanduser('~/.ros/follow_anywhere/tensorrt_cache')
TENSORRT_PROVIDER_OPTIONS = {'trt_fp16_enable': False, 'trt_engine_cache_enable': True,
                             'trt_engine_cache_path': TENSORRT_CACHE_FOLDER}
CUDA_PROVIDER = 'CUDAExecutionProvider'
CUDA_PROVIDER_OPTIONS = {'cudnn_conv_algo_search': 'HEURISTIC'}
GPU_BATCH_SIZE = 4
CPU_PROVIDER = 'CPUExecutionProvider'
NO_CUDA_PROVIDER_WARNING = 'onnxruntime has no CUDA provider (CPU build or no GPU), ReID runs on the CPU'

# Visible cells
CELL_DEPTH_MASK = 'cells'
KEYPOINT_ROW_MARGIN = 1


@dataclass
class PersonDescription:
    cellFeatures: np.ndarray
    visibleCells: np.ndarray


@dataclass
class DescriptionStack:
    cellFeatures: np.ndarray
    visibleCells: np.ndarray


def SelectExecutionProviders(requestedDevice):
    if requestedDevice not in ('auto', 'cuda'):
        return [CPU_PROVIDER], ''
    availableProviders = onnxruntime.get_available_providers()
    if CUDA_PROVIDER not in availableProviders:
        return [CPU_PROVIDER], NO_CUDA_PROVIDER_WARNING
    executionProviders = [(CUDA_PROVIDER, CUDA_PROVIDER_OPTIONS), CPU_PROVIDER]
    if TENSORRT_PROVIDER in availableProviders:
        os.makedirs(TENSORRT_CACHE_FOLDER, exist_ok=True)
        executionProviders.insert(0, (TENSORRT_PROVIDER, TENSORRT_PROVIDER_OPTIONS))
    return executionProviders, ''


class ReidModel:
    def __init__(self, backbonePath, requestedDevice=REQUESTED_DEVICE, reidThreads=REID_THREADS, reidSettings=None):
        self.settings = SimpleNamespace(**{**REID_SETTING_DEFAULTS, **(reidSettings or {})})
        executionProviders, self.deviceWarning = SelectExecutionProviders(requestedDevice)
        sessionOptions = onnxruntime.SessionOptions()
        sessionOptions.intra_op_num_threads = reidThreads
        self.session = onnxruntime.InferenceSession(backbonePath, sessionOptions, providers=executionProviders)
        if executionProviders[0] != CPU_PROVIDER and CUDA_PROVIDER not in self.session.get_providers():
            self.deviceWarning = NO_CUDA_PROVIDER_WARNING
        self.fixedBatchSize = GPU_BATCH_SIZE if CUDA_PROVIDER in self.session.get_providers() else None
        self.inputName = self.session.get_inputs()[0].name
        self.inputHeight, self.inputWidth = self.session.get_inputs()[0].shape[2:4]
        self.featureSize, self.cellRows, self.cellColumns = self.session.get_outputs()[0].shape[1:4]
        self.cellCount = self.cellRows * self.cellColumns
        backboneHead = np.load(backbonePath.replace(BACKBONE_FILE_ENDING, HEAD_FILE_ENDING))
        self.headBias = backboneHead['bias'].astype(np.float32)
        self.modelTag = f'{os.path.basename(backbonePath)}:projected:{self.settings.reidDepthMask}'
        if self.fixedBatchSize is not None:
            self.RunBackbone(np.zeros((self.fixedBatchSize, 3, self.inputHeight, self.inputWidth), np.float32))

    def DescribePeople(self, imageBgr, depthMeters, people, torsoDepths):
        if not people:
            return []
        cropBounds = [FindCropBounds(imageBgr.shape, person) for person in people]
        inputTensor = np.stack([PrepareCrop(imageBgr, bounds, self.inputWidth, self.inputHeight)
                                for bounds in cropBounds])
        featureMaps = self.RunBackbone(inputTensor)
        cellFeatures = featureMaps.transpose(0, 2, 3, 1).reshape(len(people), self.cellCount, self.featureSize)
        return [PersonDescription(personCells, self.FindVisibleCells(person, bounds, depthMeters, torsoDepth))
                for personCells, person, bounds, torsoDepth in zip(cellFeatures, people, cropBounds, torsoDepths)]

    def RunBackbone(self, inputTensor):
        if self.fixedBatchSize is None:
            return self.session.run(None, {self.inputName: inputTensor})[0]
        paddingCount = -len(inputTensor) % self.fixedBatchSize
        paddedTensor = np.concatenate([inputTensor, np.zeros((paddingCount, *inputTensor.shape[1:]), np.float32)])
        featureMaps = [self.session.run(None, {self.inputName: paddedTensor[start:start + self.fixedBatchSize]})[0]
                       for start in range(0, len(paddedTensor), self.fixedBatchSize)]
        return np.concatenate(featureMaps)[:len(inputTensor)]

    def FindVisibleCells(self, person, cropBounds, depthMeters, torsoDepth):
        left, top, right, bottom = cropBounds
        visibleCells = np.ones((self.cellRows, self.cellColumns), np.float32)
        if self.settings.reidDepthMask == CELL_DEPTH_MASK and depthMeters is not None and torsoDepth is not None:
            cropDepths = depthMeters[top:bottom, left:right]
            if cropDepths.size:
                keptPixels = ((cropDepths <= 0.0) | (np.abs(cropDepths - torsoDepth) <= self.settings.depthMaskBand))
                visibleCells = cv2.resize(keptPixels.astype(np.float32), (self.cellColumns, self.cellRows),
                                          interpolation=cv2.INTER_AREA)
        visibleCells[FindHiddenRows(person, top, bottom, self.settings.keypointConfidence, self.cellRows)] = 0.0
        return visibleCells.reshape(self.cellCount)

    def CompareDescriptions(self, query, storedStack, minimumSharedArea):
        sharedWeights = query.visibleCells[None] * storedStack.visibleCells
        visibleArea = np.maximum(query.visibleCells.sum(), storedStack.visibleCells.sum(axis=1))
        sharedAreas = sharedWeights.sum(axis=1) / np.maximum(visibleArea, 1e-9)
        queryEmbeddings = self.EmbedCells(sharedWeights @ query.cellFeatures, sharedWeights)
        storedEmbeddings = self.EmbedCells(np.einsum('vc,vcf->vf', sharedWeights, storedStack.cellFeatures),
                                           sharedWeights)
        distances = np.linalg.norm(queryEmbeddings - storedEmbeddings, axis=1)
        distances[sharedAreas < minimumSharedArea] = np.nan
        return distances, sharedAreas

    def EmbedCells(self, weightedCellSums, cellWeights):
        weightSums = np.maximum(cellWeights.sum(axis=1, keepdims=True), 1e-9)
        embeddings = np.maximum(weightedCellSums / weightSums + self.headBias, 0.0)
        return embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9)


def FindCropBounds(imageShape, person):
    imageHeight, imageWidth = imageShape[:2]
    left, top = max(0, int(person.left)), max(0, int(person.top))
    right, bottom = min(imageWidth, int(round(person.right))), min(imageHeight, int(round(person.bottom)))
    return left, top, right, bottom


def PrepareCrop(imageBgr, cropBounds, inputWidth, inputHeight):
    left, top, right, bottom = cropBounds
    cropRgb = cv2.cvtColor(imageBgr[top:bottom, left:right], cv2.COLOR_BGR2RGB)
    resizedCrop = cv2.resize(cropRgb, (inputWidth, inputHeight))
    return (resizedCrop.astype(np.float32) / 255.0).transpose(2, 0, 1)


def FindHiddenRows(person, top, bottom, keypointConfidence, cellRows):
    visibleKeypoints = person.bodyKeypoints[person.bodyKeypoints[:, 2] >= keypointConfidence]
    if not len(visibleKeypoints) or bottom <= top:
        return np.zeros(cellRows, bool)
    cellHeight = (bottom - top) / cellRows
    lowestVisibleRow = int((visibleKeypoints[:, 1].max() - top) / cellHeight)
    return np.arange(cellRows) > lowestVisibleRow + KEYPOINT_ROW_MARGIN


def StackDescriptions(descriptions):
    if not descriptions:
        return DescriptionStack(np.zeros((0, 0, 0), np.float32), np.zeros((0, 0), np.float32))
    return DescriptionStack(np.stack([description.cellFeatures for description in descriptions]),
                            np.stack([description.visibleCells for description in descriptions]))
