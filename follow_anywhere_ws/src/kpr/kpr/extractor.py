"""Public API: person crops (+ keypoints) in, part embeddings and part visibility out."""
import logging

import numpy as np
import torch

from kpr.load import load_checkpoint, load_weights
from kpr.model import KprNet, Settings
from kpr.prompt import prepare_image, prompt_masks

log = logging.getLogger('kpr')


def pick_device(requested='auto'):
    """'auto' or 'cuda' -> the GPU if there is one; otherwise warn loudly and use the CPU."""
    if requested in ('auto', 'cuda'):
        if torch.cuda.is_available():
            return 'cuda'
        log.warning('No CUDA GPU found: KPR is switching to CPU, which is slow')
    return 'cpu'


class Kpr:
    """KPR embedder. Rows of the output: 0 whole person, then head, left arm, right arm, torso,
    left leg, right leg, left foot, right foot (see kpr.prompt.PART_NAMES)."""

    def __init__(self, weights, device='auto'):
        self.device = pick_device(device)
        checkpoint = load_checkpoint(weights)
        self.settings = Settings.from_config(checkpoint.get('config', {}))
        self.net = KprNet(self.settings)
        load_weights(self.net, checkpoint)
        self.net.eval().to(self.device)

    @torch.no_grad()
    def __call__(self, crops, keypoints=None, negatives=None):
        """Describe person crops.

        crops: list of RGB uint8 images (h, w, 3).
        keypoints: per crop the person's own COCO-17 (17, 3) xyc in crop pixels, or None for no
            prompt (a list of None, or None). Drop low-confidence points before passing them.
        negatives: per crop a list of (17, 3) skeletons of OTHER people (negative prompts).
        Returns embeddings (N, 9, 512) float32, L2-normalised, and visible (N, 9) bool.
        """
        s = self.settings
        n = len(crops)
        keypoints = keypoints or [None] * n
        negatives = negatives or [[] for _ in range(n)]
        images = torch.stack([prepare_image(c, s.height, s.width) for c in crops])
        prompts = torch.stack([
            prompt_masks(k, neg, c.shape[:2], s.height, s.width, s.vis_thresh, s.use_negative)
            for c, k, neg in zip(crops, keypoints, negatives)])
        emb, visible = self.net(images.to(self.device), prompts.to(self.device))
        return emb.cpu().numpy().astype(np.float32), visible.cpu().numpy()
