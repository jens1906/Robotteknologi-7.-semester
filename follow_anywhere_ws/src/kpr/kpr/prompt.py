"""Person crop + keypoints -> the image and prompt tensors KPR takes.

Mirrors what upstream's ImageDataset.getitem does at test time: resize, normalise, rescale the
keypoints, draw a Gaussian blob per keypoint, group them into 8 body parts ("cck8"), add the
negative-keypoint channel and a background channel.

Keypoints are COCO-17 (x, y, confidence) in crop pixels, the layout YOLO-pose outputs.
"""
import cv2
import numpy as np
import torch

# the 8 prompt parts as COCO-17 indices: head, left arm, right arm, torso, left leg, right leg, feet
PARTS = ((0, 1, 2, 3, 4), (5, 7, 9), (6, 8, 10), (5, 6, 11, 12), (13, 15), (14, 16), (15,), (16,))
PART_NAMES = ('head', 'left arm', 'right arm', 'torso', 'left leg', 'right leg',
              'left foot', 'right foot')
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def prepare_image(rgb, height, width):
    """RGB uint8 crop -> normalised float tensor (3, height, width)."""
    img = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    img = (img.astype(np.float32) / 255.0 - _MEAN) / _STD
    return torch.from_numpy(img.transpose(2, 0, 1).copy())


def fit_keypoints(kps, crop_hw, height, width):
    """(17, 3) xyc in crop pixels -> network input pixels. Points outside the crop become invisible."""
    kps = np.nan_to_num(np.asarray(kps, dtype=np.float64).reshape(17, 3))
    kps[:, 0] *= width / crop_hw[1]
    kps[:, 1] *= height / crop_hw[0]
    outside = ((kps[:, 0] < 0) | (kps[:, 0] >= width) | (kps[:, 1] < 0) | (kps[:, 1] >= height)
               | (kps[:, 2] <= 0))
    kps[outside] = 0
    return kps


def _blob(width):
    """Gaussian bump, radius width/11 (upstream KeypointsToMasks, g_scale=11)."""
    radius = int(width / 11)
    n = np.arange(2 * radius + 1) - radius
    g = np.exp(-0.5 * (n / ((2 * radius + 1) / 4)) ** 2)
    return np.outer(g, g)


def heatmaps(kps, height, width, vis_thresh):
    """One Gaussian blob per keypoint with confidence above vis_thresh: (17, height, width)."""
    out = np.zeros((len(kps), height, width), dtype=np.float32)
    blob = _blob(width)
    rad = blob.shape[0] // 2
    for i, (x, y, conf) in enumerate(kps):
        if conf <= vis_thresh:
            continue
        x, y = int(x), int(y)
        up, down = min(rad, y), min(rad, height - 1 - y)
        left, right = min(rad, x), min(rad, width - 1 - x)
        out[i, y - up:y + down + 1, x - left:x + right + 1] = \
            blob[rad - up:rad + down + 1, rad - left:rad + right + 1]
    return out


def prompt_masks(target, negatives, crop_hw, height, width, vis_thresh=0.3, use_negative=True):
    """Prompt tensor (1 + use_negative + 8, height, width): [background, negative, 8 parts].

    target: (17, 3) keypoints of the person to describe, or None for no prompt (all zeros, which
    is the 'empty prompt' the network saw in training). negatives: (17, 3) skeletons of other people.
    """
    if target is None:
        target = np.zeros((17, 3))
    maps = heatmaps(fit_keypoints(target, crop_hw, height, width), height, width, vis_thresh)
    chans = np.stack([maps[list(p)].max(axis=0) for p in PARTS]).clip(0, 1)
    if use_negative:
        neg = [heatmaps(fit_keypoints(k, crop_hw, height, width), height, width, vis_thresh).max(axis=0)
               for k in negatives or []]
        neg = np.max(neg, axis=0) if neg else np.zeros((height, width), dtype=np.float32)
        chans = np.concatenate([neg[None], chans])
    background = (1 - chans.sum(axis=0)).clip(0, 1)
    return torch.from_numpy(np.concatenate([background[None], chans]).astype(np.float32))
