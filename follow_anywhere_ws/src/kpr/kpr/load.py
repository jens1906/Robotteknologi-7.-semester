"""Loading a KPR checkpoint. It is a pickle from the internet, so it is read with weights_only=True.

Besides tensors the file holds the training config (a yacs CfgNode, mapped to a plain OrderedDict),
a few saved metrics (numpy scalars) and a python-2 style set. Exactly those globals are allowed
(found with pickletools, which does not execute anything); anything else is refused.
Needs a torch whose safe_globals() accepts (callable, 'dotted.name') pairs (2.14 here).
"""
import codecs
from collections import OrderedDict

import numpy as np
import torch

try:
    from numpy._core.multiarray import scalar as np_scalar    # numpy 2
except ImportError:
    from numpy.core.multiarray import scalar as np_scalar     # numpy 1

# checkpoint keys that only exist for training, or that the test-time forward never uses
UNUSED_PREFIXES = (
    'global_identity_classifier.', 'background_identity_classifier.',
    'concat_parts_identity_classifier.', 'parts_identity_classifier.',
    'foreground_identity_classifier.classifier.',
    'global_after_pooling_dim_reduce.', 'background_after_pooling_dim_reduce.',
    'backbone_appearance_feature_extractor.norm.',
    'backbone_appearance_feature_extractor.parts_embed',
)


def _allowed_globals():
    dtypes = [getattr(np.dtypes, n) for n in ('Float32DType', 'Float64DType', 'Int32DType', 'Int64DType')]
    return [(OrderedDict, 'yacs.config.CfgNode'), (set, '__builtin__.set'),
            (codecs.encode, '_codecs.encode'), (np_scalar, 'numpy.core.multiarray.scalar'),
            np.dtype, *dtypes]


def load_checkpoint(path):
    """The checkpoint dict: 'state_dict', 'config' (nested OrderedDicts) and training leftovers."""
    try:
        with torch.serialization.safe_globals(_allowed_globals()):
            return torch.load(path, map_location='cpu', weights_only=True)
    except Exception as e:
        raise RuntimeError(f'{path}: the safe checkpoint loader refused or failed: {e}') from e


def load_weights(net, checkpoint):
    """Strict load: every key the network has must be in the checkpoint, and every checkpoint key
    must be used or be on the UNUSED_PREFIXES list. Shapes must match."""
    state = checkpoint.get('state_dict', checkpoint)
    state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
    wanted = set(net.state_dict())
    missing = sorted(wanted - set(state))
    extra = sorted(k for k in set(state) - wanted if not k.startswith(UNUSED_PREFIXES))
    if missing or extra:
        raise RuntimeError(f'checkpoint does not fit the network. Missing {len(missing)}: '
                           f'{missing[:6]} ... Unexpected {len(extra)}: {extra[:6]} ...')
    net.load_state_dict({k: state[k] for k in wanted})
