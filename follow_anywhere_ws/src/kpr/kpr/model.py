"""The KPR network at test time (timm SwinV2 variant, learned part attention, binary visibility).

Module and parameter names are upstream's (torchreid/models/kpr.py, promptable_timm_swin.py,
promptable_transformer_backbone.py), so a checkpoint loads with a strict key check.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

# What this trimmed copy supports. The checkpoint's own config is checked against it, so a
# checkpoint made with other settings fails loudly instead of giving wrong embeddings.
SUPPORTED = {
    'model.kpr.keypoints.prompt_masks': 'keypoints_gaussian',
    'model.kpr.keypoints.prompt_preprocess': 'cck8',
    'model.kpr.keypoints.vis_continous': False,
    'model.promptable_trans.masks_prompting': True,
    'model.promptable_trans.pose_encoding_strategy': 'embed_heatmaps_patches',
    'model.promptable_trans.pose_encoding_all_layers': False,
    'model.promptable_trans.disable_inference_prompting': False,
    'model.kpr.learnable_attention_enabled': True,
    'model.kpr.test_use_target_segmentation': 'none',
    'model.kpr.testing_binary_visibility_score': True,
    'model.kpr.use_prompt_visibility_score': False,
    'model.kpr.enable_fpn': False,
    'model.kpr.enable_msf': True,
    'model.kpr.msf_spatial_scale': -1,
    'model.kpr.dim_reduce': 'after_pooling',
    'model.kpr.pooling': 'gwap',
    'model.kpr.normalization': 'identity',
    'model.kpr.test_embeddings': ['bn_foreg', 'parts'],
    'test.normalize_feature': True,
}
# upstream defaults (torchreid/scripts/default_config.py) for keys a checkpoint config lacks
DEFAULTS = {
    **SUPPORTED,
    'model.kpr.backbone': 'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k',
    'model.kpr.masks.parts_num': 8,
    'model.kpr.dim_reduce_output': 512,
    'model.kpr.keypoints.use_negative_keypoints': True,
    'model.kpr.keypoints.vis_thresh': 0.3,
    'model.promptable_trans.no_background_token': False,
    'data.height': 256,
    'data.width': 128,
}


def _dig(cfg, path, default):
    for key in path.split('.'):
        if not isinstance(cfg, dict) or key not in cfg:
            return default
        cfg = cfg[key]
    return cfg


@dataclass
class Settings:
    backbone: str
    height: int
    width: int
    parts: int
    out_dim: int
    use_negative: bool
    drop_background: bool
    vis_thresh: float

    @classmethod
    def from_config(cls, cfg):
        """cfg: the config dict stored in the checkpoint (may be empty: upstream defaults then)."""
        for path, wanted in SUPPORTED.items():
            got = _dig(cfg, path, DEFAULTS[path])
            if got != wanted:
                raise NotImplementedError(
                    f'checkpoint has {path} = {got!r}, this KPR copy only supports {wanted!r}')
        get = lambda path: _dig(cfg, path, DEFAULTS[path])  # noqa: E731
        if not get('model.kpr.backbone').startswith('swin'):
            raise NotImplementedError(f'backbone {get("model.kpr.backbone")!r}: only timm swin '
                                      '(the ImageNet KPR weights) is supported')
        return cls(get('model.kpr.backbone'), get('data.height'), get('data.width'),
                   get('model.kpr.masks.parts_num'), get('model.kpr.dim_reduce_output'),
                   get('model.kpr.keypoints.use_negative_keypoints'),
                   get('model.promptable_trans.no_background_token'),
                   get('model.kpr.keypoints.vis_thresh'))

    @property
    def prompt_channels(self):
        """Background (unless dropped), negative keypoints (if used) and the 8 prompt parts."""
        return 8 + (not self.drop_background) + self.use_negative


class PromptedSwin(nn.Module):
    """timm Swin whose patch tokens get the keypoint heatmaps added (upstream SwinTransformer)."""

    def __init__(self, name, img_size, prompt_channels, drop_background):
        super().__init__()
        import timm
        from timm.layers import PatchEmbed
        model = timm.create_model(name, pretrained=False, num_classes=0, global_pool='',
                                  img_size=img_size)
        self.patch_embed = model.patch_embed
        self.masks_patch_embed = PatchEmbed(
            in_chans=prompt_channels, img_size=img_size, patch_size=model.patch_embed.patch_size,
            embed_dim=model.embed_dim,
            norm_layer=(None if isinstance(model.patch_embed.norm, nn.Identity)
                        else model.patch_embed.norm.__class__),
            output_fmt='NHWC')
        self.model = model
        self.drop_background = drop_background
        self.stage_channels = [info['num_chs'] for info in model.feature_info]

    def forward(self, images, prompt):
        x = self.patch_embed(images)                                  # (N, H/4, W/4, C)
        if self.drop_background:
            prompt = prompt[:, 1:]
        x = x + self.masks_patch_embed(prompt.to(x.dtype))
        stages = []
        for layer in self.model.layers:
            x = layer(x)
            stages.append(x.permute(0, 3, 1, 2))                      # (N, C, h, w)
        return stages


class DimReduce2d(nn.Module):
    """1x1 conv + BN + ReLU (upstream BeforePoolingDimReduceLayer)."""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(n_in, n_out, 1), nn.BatchNorm2d(n_out),
                                    nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layers(x)


class MultiStageFusion(nn.Module):
    """Resize every Swin stage to the first stage's size, concatenate, reduce the channels."""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.dim_reduce = DimReduce2d(n_in, n_out)

    def forward(self, stages):
        size = stages[0].shape[2:]
        maps = [stages[0]] + [F.interpolate(s, size=size, mode='bilinear', align_corners=True)
                              for s in stages[1:]]
        return self.dim_reduce(torch.cat(maps, 1))


class AfterPoolingDimReduce(nn.Module):
    """Linear + BN + ReLU on pooled vectors, (N, C) or (N, K, C) (upstream AfterPoolingDimReduceLayer)."""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(n_in, n_out), nn.BatchNorm1d(n_out),
                                    nn.ReLU(inplace=True))

    def forward(self, x):
        if x.dim() == 3:
            n, k, _ = x.shape
            return self.layers(x.flatten(0, 1)).view(n, k, -1)
        return self.layers(x)


class PixelToPart(nn.Module):
    """Learned attention: every feature-map pixel -> background or one of the K parts."""

    def __init__(self, n_in, parts):
        super().__init__()
        self.bn = nn.BatchNorm2d(n_in)
        self.classifier = nn.Conv2d(n_in, parts + 1, 1)

    def forward(self, x):
        return self.classifier(self.bn(x))


class BatchNormOnly(nn.Module):
    """The BN half of upstream's BNClassifier. The identity classifier itself is training only."""

    def __init__(self, n):
        super().__init__()
        self.bn = nn.BatchNorm1d(n)

    def forward(self, x):
        return self.bn(x)


class KprNet(nn.Module):
    def __init__(self, s: Settings):
        super().__init__()
        self.parts = s.parts
        self.backbone_appearance_feature_extractor = PromptedSwin(
            s.backbone, (s.height, s.width), s.prompt_channels, s.drop_background)
        channels = self.backbone_appearance_feature_extractor.stage_channels
        self.msf = MultiStageFusion(sum(channels), channels[-1])
        self.pixel_classifier = PixelToPart(channels[-1], s.parts)
        self.foreground_after_pooling_dim_reduce = AfterPoolingDimReduce(channels[-1], s.out_dim)
        self.parts_after_pooling_dim_reduce = AfterPoolingDimReduce(channels[-1], s.out_dim)
        self.foreground_identity_classifier = BatchNormOnly(s.out_dim)

    def forward(self, images, prompt):
        """images (N, 3, H, W), prompt (N, C, H, W) -> embeddings (N, 1 + K, D), visible (N, 1 + K).

        Row 0 is the whole person, rows 1..K the body parts. Embeddings are L2-normalised.
        """
        stages = self.backbone_appearance_feature_extractor(images, prompt)
        feats = self.msf(stages)                                           # (N, C, h, w)
        probs = F.softmax(self.pixel_classifier(feats), dim=1)             # (N, K+1, h, w)
        part_masks = probs[:, 1:]
        person_mask = part_masks.max(dim=1)[0]

        # binary visibility as upstream: a part is visible if it owns at least one pixel
        owner = F.one_hot(probs.argmax(dim=1), self.parts + 1)             # (N, h, w, K+1)
        seen = owner.amax(dim=(1, 2)).bool()                               # (N, K+1)
        visible = torch.cat([seen.amax(dim=1, keepdim=True), seen[:, 1:]], dim=1)

        # mask-weighted pooling; the einsum equals upstream's multiply-then-pool without a K x C x h x w tensor
        person = torch.einsum('nhw,nchw->nc', person_mask, feats) / person_mask[0].numel()
        mass = part_masks.sum(dim=(2, 3)).clamp(min=1e-6)
        parts = torch.einsum('nkhw,nchw->nkc', part_masks, feats) / mass[..., None]

        person = self.foreground_identity_classifier(self.foreground_after_pooling_dim_reduce(person))
        parts = self.parts_after_pooling_dim_reduce(parts)
        return F.normalize(torch.cat([person[:, None], parts], dim=1), dim=-1), visible
