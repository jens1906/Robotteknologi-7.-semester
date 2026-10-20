# Notice

This package is a re-implementation of the **test-time path** of
[Keypoint Promptable Re-Identification](https://github.com/VlSomers/keypoint_promptable_reidentification)
(Somers, Alahi, De Vleeschouwer, ECCV 2024), written against upstream commit `e3e6ee2`
(2025-06-12). It is a derived work and is distributed under upstream's
Hippocratic License 3.0 (`LICENSE`). Upstream's `setup.py` says MIT while its `LICENSE` file is
Hippocratic 3.0; we follow the stricter `LICENSE` file.

What it is: the network and the keypoint-prompt preprocessing needed to turn a person crop plus
keypoints into part embeddings and part visibility, for the released **timm SwinV2-B (ImageNet)**
checkpoints (`configs/kpr/imagenet/*`, 256x128, 8 body parts).

Module and parameter names follow upstream so the released checkpoints load with a strict key check.
The only keys left out are the ones that exist for training (identity classifiers, unused embeddings).

What was left out on purpose: training, losses, datasets, evaluation, augmentation (BIPO),
the SOLIDER Swin backbone, FPN, target-mask (human parsing) inputs, every other backbone, and the
`torchreid` package itself (its `__init__` imports the whole training stack).

Upstream detail worth knowing: its `demo.py` feeds BGR images, while training read RGB
(`torchreid/utils/tools.py: read_image`). This package takes RGB.
