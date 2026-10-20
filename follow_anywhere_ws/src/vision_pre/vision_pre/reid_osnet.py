"""OSNet person re-identification, the old baseline. Opt-in (embedder 'osnet'), everything it needs is here.

A 2.2 M parameter CNN, one 512-d vector per crop, about 5 ms on a laptop CPU. The ONNX model is
downloaded once to ~/.cache/follow_anywhere/osnet. Needs onnxruntime in the python environment.
Same call as kpr.Kpr, but it ignores keypoints and returns one 'part' (the whole person).
"""
import os
import urllib.request
import zipfile

import cv2
import numpy as np

URL = ('https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/'
       'osnet/releases/v0.64.0/osnet-onnx-float.zip')
FOLDER = '~/.cache/follow_anywhere/osnet'
FILES = ('osnet.onnx', 'osnet.data')          # the .data file holds the weights


def _download():
    folder = os.path.expanduser(FOLDER)
    if all(os.path.isfile(os.path.join(folder, f)) for f in FILES):
        return os.path.join(folder, 'osnet.onnx')
    os.makedirs(folder, exist_ok=True)
    archive = os.path.join(folder, 'download.zip')
    urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if os.path.basename(name) in FILES:                    # only the two files, flat
                with z.open(name) as src, open(os.path.join(folder, os.path.basename(name)), 'wb') as dst:
                    dst.write(src.read())
    os.remove(archive)
    return os.path.join(folder, 'osnet.onnx')


class Osnet:
    def __init__(self, threads=2):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads                       # leave cores for the detector
        self.session = ort.InferenceSession(_download(), opts, providers=['CPUExecutionProvider'])

    def __call__(self, crops, keypoints=None, negatives=None):
        """crops: RGB uint8 images. Returns embeddings (N, 1, 512) and visible (N, 1)."""
        out = []
        for crop in crops:
            # the ImageNet mean/std normalisation is part of the model: pass plain 0..1 RGB
            x = (cv2.resize(crop, (128, 256)).astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
            vec = self.session.run(None, {'image': x})[0][0]
            out.append(vec / max(float(np.linalg.norm(vec)), 1e-9))
        emb = np.stack(out).astype(np.float32)[:, None]
        return emb, np.ones(emb.shape[:2], bool)
