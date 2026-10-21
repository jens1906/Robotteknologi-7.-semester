"""Describes how a person looks (re-identification). Plain python, no ROS. OSNet.

To include:
- Input: the colour image and the people found in it. Each person is cut out by their box and turned into
  one 512-number vector (embedding) of unit length. The same person gives close vectors, different people
  far ones: the distance between two embeddings is 0 (same) to 2 (opposite).
- OSNet: ONNX model run with onnxruntime, loaded from the package's resource/models folder (fetch it with
  scripts/fetch_models.py). GPU (CUDA provider) when onnxruntime has it, else CPU with a warning.
  Measured: ~14 ms per crop on a laptop CPU with 2 threads.
- Only computes what it is asked for: the decision "is this the leader" is in leader_manager.
- Known OSNet weakness: two people overlapping in one box give a mixed description, so the leader is
  not compared while someone overlaps them.
"""

import cv2
import numpy as np
import onnxruntime

# Parameter defaults (set in config/perception_params.yaml)
REQUESTED_DEVICE = 'auto'
REID_THREADS = 2

# OSNet model
OSNET_INPUT_NAME = 'image'
OSNET_INPUT_WIDTH = 128
OSNET_INPUT_HEIGHT = 256
OSNET_EMBEDDING_SIZE = 512
CUDA_PROVIDER = 'CUDAExecutionProvider'
CPU_PROVIDER = 'CPUExecutionProvider'
NO_CUDA_PROVIDER_WARNING = 'onnxruntime has no CUDA provider (CPU build or no GPU), ReID runs on the CPU'


def SelectExecutionProviders(requestedDevice):
    if requestedDevice not in ('auto', 'cuda'):
        return [CPU_PROVIDER], ''
    if CUDA_PROVIDER in onnxruntime.get_available_providers():
        return [CUDA_PROVIDER, CPU_PROVIDER], ''
    return [CPU_PROVIDER], NO_CUDA_PROVIDER_WARNING


class ReidModel:
    def __init__(self, reidModelPath, requestedDevice=REQUESTED_DEVICE, reidThreads=REID_THREADS):
        executionProviders, self.deviceWarning = SelectExecutionProviders(requestedDevice)
        sessionOptions = onnxruntime.SessionOptions()
        sessionOptions.intra_op_num_threads = reidThreads
        self.session = onnxruntime.InferenceSession(reidModelPath, sessionOptions, providers=executionProviders)
        if CUDA_PROVIDER in executionProviders and CUDA_PROVIDER not in self.session.get_providers():
            self.deviceWarning = NO_CUDA_PROVIDER_WARNING

    def EmbedPeople(self, imageBgr, detectedPeople):
        if not detectedPeople:
            return np.zeros((0, OSNET_EMBEDDING_SIZE), np.float32)
        return np.stack([self.EmbedCrop(self.CutOutPerson(imageBgr, person)) for person in detectedPeople])

    def CutOutPerson(self, imageBgr, person):
        imageHeight, imageWidth = imageBgr.shape[:2]
        left, top = max(0, int(person.left)), max(0, int(person.top))
        right, bottom = min(imageWidth, int(round(person.right))), min(imageHeight, int(round(person.bottom)))
        return imageBgr[top:bottom, left:right]

    def EmbedCrop(self, cropBgr):
        cropRgb = cv2.cvtColor(cropBgr, cv2.COLOR_BGR2RGB)
        resizedCrop = cv2.resize(cropRgb, (OSNET_INPUT_WIDTH, OSNET_INPUT_HEIGHT))
        inputTensor = (resizedCrop.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        embedding = self.session.run(None, {OSNET_INPUT_NAME: inputTensor})[0][0]
        return embedding / max(float(np.linalg.norm(embedding)), 1e-9)
