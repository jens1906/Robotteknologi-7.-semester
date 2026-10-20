"""Tests that need no weights and no timm."""
import sys
import types

import numpy as np
import pytest
import torch

from kpr import part_distance
from kpr.load import UNUSED_PREFIXES, load_checkpoint, load_weights
from kpr.model import Settings
from kpr.prompt import fit_keypoints, heatmaps, prompt_masks

H, W = 256, 128


def skeleton(conf=1.0):
    """A standing person in a 100 x 200 crop, COCO-17 xyc."""
    pts = [(50, 20), (45, 15), (55, 15), (40, 18), (60, 18), (35, 50), (65, 50), (30, 90),
           (70, 90), (28, 120), (72, 120), (40, 110), (60, 110), (40, 150), (60, 150),
           (40, 190), (60, 190)]
    return np.array([(x, y, conf) for x, y in pts], dtype=float)


def test_prompt_layout_and_empty_prompt():
    p = prompt_masks(skeleton(), [], (200, 100), H, W)
    assert p.shape == (10, H, W)                          # background, negative, 8 parts
    assert p[1].sum() == 0                                # no negative skeleton given
    assert p[2:].max() == pytest.approx(1.0)
    empty = prompt_masks(None, [], (200, 100), H, W)
    assert empty[0].min() == 1 and empty[1:].sum() == 0   # the "no prompt" the network trained with


def test_negative_skeleton_and_low_confidence():
    p = prompt_masks(skeleton(), [skeleton()], (200, 100), H, W)
    assert p[1].max() > 0
    low = prompt_masks(skeleton(conf=0.2), [], (200, 100), H, W)      # below vis_thresh 0.3
    assert low[2:].sum() == 0


def test_keypoints_outside_the_crop_are_dropped():
    kps = skeleton()
    kps[0, :2] = (150, 20)                                # right of a 100 px wide crop
    fitted = fit_keypoints(kps, (200, 100), H, W)
    assert (fitted[0] == 0).all() and fitted[1, 2] == 1
    assert heatmaps(fitted, H, W, 0.3)[0].sum() == 0


def test_part_distance_uses_only_shared_parts_and_flags_unknown():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(9, 16))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    same, other = q.copy(), -q
    gallery = np.stack([same, other, same])
    q_vis = np.ones(9, bool)
    g_vis = np.stack([np.ones(9, bool), np.ones(9, bool), np.array([1, 1, 0, 0, 0, 0, 0, 0, 0], bool)])
    d = part_distance(q, q_vis, gallery, g_vis)
    assert d[0] == pytest.approx(0) and d[1] == pytest.approx(2)
    assert np.isnan(d[2])                                 # only one body part shared: unknown
    assert not np.isnan(part_distance(q, q_vis, gallery[:1], g_vis[:1], min_parts=2)[0])


def test_part_distance_with_a_single_whole_person_vector():
    q = np.array([[1.0, 0.0]])
    g = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
    d = part_distance(q, np.ones(1, bool), g, np.ones((2, 1), bool))
    assert d[0] == pytest.approx(0) and d[1] == pytest.approx(2 ** 0.5)
    assert np.isnan(part_distance(q, np.zeros(1, bool), g, np.ones((2, 1), bool))).all()


def test_settings_rejects_other_checkpoint_configs():
    assert Settings.from_config({}).parts == 8
    with pytest.raises(NotImplementedError):
        Settings.from_config({'model': {'kpr': {'enable_fpn': True}}})
    with pytest.raises(NotImplementedError):
        Settings.from_config({'model': {'kpr': {'backbone': 'resnet50'}}})


class CfgNode(dict):
    """Looks like yacs.config.CfgNode as pickled: a dict subclass that also carries instance state."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__['__immutable__'] = False


CfgNode.__module__ = 'yacs.config'


def test_safe_loader_reads_a_yacs_style_config_and_nothing_else(tmp_path):
    fake = types.ModuleType('yacs.config')
    fake.CfgNode = CfgNode
    sys.modules['yacs'], sys.modules['yacs.config'] = types.ModuleType('yacs'), fake
    try:                                       # yacs only exists while the test checkpoint is written
        cfg = CfgNode(model=CfgNode(kpr=CfgNode(backbone='swin')))
        path = str(tmp_path / 'ok.pth.tar')
        torch.save({'state_dict': {'module.a': torch.zeros(2)}, 'config': cfg}, path)
        evil = str(tmp_path / 'evil.pth.tar')
        torch.save({'x': types.SimpleNamespace(a=1)}, evil)
    finally:
        sys.modules.pop('yacs', None)
        sys.modules.pop('yacs.config', None)
    ckpt = load_checkpoint(path)
    assert ckpt['config']['model']['kpr']['backbone'] == 'swin'
    with pytest.raises(RuntimeError):
        load_checkpoint(evil)


def test_load_weights_is_strict():
    net = torch.nn.Linear(2, 2)
    good = {'module.weight': torch.ones(2, 2), 'module.bias': torch.zeros(2),
            UNUSED_PREFIXES[0] + 'w': torch.zeros(1)}
    load_weights(net, {'state_dict': good})
    assert net.weight.sum() == 4
    with pytest.raises(RuntimeError):
        load_weights(net, {'state_dict': {'weight': torch.ones(2, 2)}})                # bias missing
    with pytest.raises(RuntimeError):
        load_weights(net, {'state_dict': {**good, 'surprise': torch.zeros(1)}})        # unknown key
