import os

import numpy as np
import pytest

from perception import person_detector
from perception.person_detector import NO_CUDA_GPU_WARNING, PersonDetector, SelectInferenceDevice

MODEL_FILE = 'yolov8n-pose.pt'


def test_cpu_request_gives_cpu_without_warning():
    assert SelectInferenceDevice('cpu') == ('cpu', '')


def test_auto_without_cuda_falls_back_to_cpu_with_warning(monkeypatch):
    monkeypatch.setattr(person_detector.torch.cuda, 'is_available', lambda: False)
    assert SelectInferenceDevice('auto') == ('cpu', NO_CUDA_GPU_WARNING)


def test_auto_with_cuda_uses_the_gpu(monkeypatch):
    monkeypatch.setattr(person_detector.torch.cuda, 'is_available', lambda: True)
    assert SelectInferenceDevice('auto') == ('cuda:0', '')


@pytest.mark.skipif(not os.path.exists(MODEL_FILE), reason=f'{MODEL_FILE} not in the working directory')
def test_blank_image_has_no_people():
    personDetector = PersonDetector(MODEL_FILE, requestedDevice='cpu')
    assert personDetector.DetectAndTrackPeople(np.zeros((480, 640, 3), np.uint8)) == []
