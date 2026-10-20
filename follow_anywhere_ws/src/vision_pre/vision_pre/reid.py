"""People -> embeddings. make_embedder() returns embed(frame_bgr, targets, people) -> (emb, visible).

emb is (N, P, D) L2-normalised, visible (N, P) bool, one row per target person. P is 9 for KPR
(whole person + 8 body parts) and 1 for OSNet. Distances between them: kpr.part_distance.
"""
import glob
import os

import cv2
import numpy as np

KPR_WEIGHTS = 'kpr_occ_pt_IN_*.pth.tar'

# Leader thresholds per embedder (distance 0..2, closest of 8 stored views). Measured on MOT17-04 and
# -09 with the first 8 views of a person as the gallery: `accept` lets about 0.2% of other people
# through per comparison and keeps about half of the true views, `drop` is above 99% of true views.
# 0.65 (my first guess) would let 2% (KPR) to 6% (OSNet) of other people through. overlap_ok: KPR
# is told about the people standing over a crop (negative prompts), OSNet is not.
THRESHOLDS = {'kpr': dict(accept=0.52, drop=0.80, overlap_ok=True),
              'osnet': dict(accept=0.45, drop=0.82, overlap_ok=False)}


def find_weights(pattern=KPR_WEIGHTS):
    """models/<pattern> in the workspace, searched upward from the working directory and from here."""
    for start in (os.getcwd(), os.path.dirname(os.path.abspath(__file__))):
        folder = start
        while True:
            hits = sorted(glob.glob(os.path.join(folder, 'models', pattern)))
            if hits:
                return hits[0]
            parent = os.path.dirname(folder)
            if parent == folder:
                break
            folder = parent
    raise FileNotFoundError(f'no models/{pattern} found: put the KPR weights in '
                            'follow_anywhere_ws/models/ or pass the path')


def _overlap(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def crop_inputs(frame_rgb, person, people):
    """(crop, own keypoints, neighbours' keypoints), the keypoints moved into crop pixels.

    Neighbours are the other people whose box overlaps this one: KPR's negative prompts.
    """
    h, w = frame_rgb.shape[:2]
    x1, y1 = max(0, int(person.box[0])), max(0, int(person.box[1]))
    x2, y2 = min(w, int(round(person.box[2]))), min(h, int(round(person.box[3])))
    shift = np.array([x1, y1, 0.0])
    neighbours = [o.kps - shift for o in people if o.id != person.id and _overlap(person.box, o.box)]
    return frame_rgb[y1:y2, x1:x2], person.kps - shift, neighbours


def make_embedder(name='kpr', weights='', device='auto', prompt=True):
    if name == 'kpr':
        from kpr import Kpr
        weights = weights or find_weights()
        net, tag = Kpr(weights, device), f'kpr:{os.path.basename(weights)}'
    elif name == 'osnet':
        from vision_pre.reid_osnet import Osnet
        net, tag = Osnet(), 'osnet'
    else:
        raise ValueError(f"unknown embedder '{name}', use 'kpr' or 'osnet'")

    def embed(frame_bgr, targets, people):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        crops, kps, negs = zip(*[crop_inputs(rgb, p, people) for p in targets])
        return net(list(crops), list(kps) if prompt else None, list(negs) if prompt else None)

    embed.tag = tag            # names the model, saved leaders are only used with the model that made them
    return embed
