"""Turns an image crop into a unit-length appearance vector.

OsnetEmbedder (default) is a model trained to tell people apart. DinoEmbedder is a general image
model that also works for non-person targets. Heavy imports are done inside the classes.
"""
import os
import urllib.request
import zipfile

import cv2
import numpy as np

OSNET_URL = ('https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/'
             'osnet/releases/v0.64.0/osnet-onnx-float.zip')
OSNET_DIR = '~/.cache/follow_anywhere/osnet'
_OSNET_FILES = ('osnet.onnx', 'osnet.data')        # the .data file holds the weights


def _ensure_osnet(path, url):
    """Return the path to osnet.onnx, downloading and unpacking the release zip once."""
    if path:
        return os.path.expanduser(path)
    folder = os.path.expanduser(OSNET_DIR)
    model = os.path.join(folder, 'osnet.onnx')
    if all(os.path.isfile(os.path.join(folder, f)) for f in _OSNET_FILES):
        return model
    os.makedirs(folder, exist_ok=True)
    archive = os.path.join(folder, 'download.zip')
    urllib.request.urlretrieve(url, archive)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            base = os.path.basename(name)
            if base in _OSNET_FILES:               # only the two expected files, flat
                with z.open(name) as src, open(os.path.join(folder, base), 'wb') as dst:
                    dst.write(src.read())
    os.remove(archive)
    return model


class OsnetEmbedder:
    """OSNet person re-identification, 512-d, about 5 ms per crop on a laptop CPU."""
    name = 'osnet'

    def __init__(self, path='', url=OSNET_URL):
        from vision.venv_fallback import ensure
        ort = ensure('onnxruntime')
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2              # leave cores for YOLO and ROS
        self.session = ort.InferenceSession(
            _ensure_osnet(path, url), opts, providers=['CPUExecutionProvider'])

    def embed(self, bgr_crop):
        """Return a unit vector (512,) describing the crop."""
        rgb = cv2.cvtColor(cv2.resize(bgr_crop, (128, 256)), cv2.COLOR_BGR2RGB)
        # the ImageNet mean/std normalisation is part of the model, so pass plain 0..1 RGB
        x = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        vec = self.session.run(None, {'image': x})[0][0]
        return vec / max(float(np.linalg.norm(vec)), 1e-9)


class DinoEmbedder:
    name = 'dino'
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    _PATCH = 14   # DINOv2 patch size, input sides must be a multiple of this

    def __init__(self, name='facebook/dinov2-small', size=224, device='cpu'):
        from transformers import AutoModel
        self.size = size
        self.device = device
        self.model = AutoModel.from_pretrained(name).to(device).eval()

    def _prepare(self, bgr):
        import torch
        # Keep the aspect ratio (a person crop is tall): longer side = size,
        # shorter side rounded to a multiple of the patch size.
        h, w = bgr.shape[:2]
        scale = self.size / max(h, w)
        nh = max(self._PATCH * 4, round(h * scale / self._PATCH) * self._PATCH)
        nw = max(self._PATCH * 4, round(w * scale / self._PATCH) * self._PATCH)
        rgb = cv2.cvtColor(cv2.resize(bgr, (nw, nh)), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - self._MEAN) / self._STD
        return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

    def embed(self, bgr_crop):
        """Return a unit vector (D,) describing the crop."""
        import torch
        with torch.no_grad():
            out = self.model(pixel_values=self._prepare(bgr_crop))
            vec = torch.nn.functional.normalize(out.last_hidden_state[:, 0], dim=1)   # CLS token
        return vec[0].cpu().numpy()


def make_embedder(name='osnet', osnet_path='', osnet_url=OSNET_URL):
    if name == 'osnet':
        return OsnetEmbedder(osnet_path, osnet_url)
    if name == 'dino':
        return DinoEmbedder()
    raise ValueError(f"unknown embedder '{name}', use 'osnet' or 'dino'")
