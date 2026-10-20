"""Save and load the leader description (embedding gallery) as an .npz file."""
import os
import time

import numpy as np


def save(path, gallery, cls, col=None, embedder=''):
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    gallery = np.asarray(gallery, dtype=np.float32)
    extra = {} if col is None else {'col': np.asarray(col, dtype=np.float32)}
    np.savez(path, gallery=gallery, mean=gallery.mean(axis=0), cls=int(cls),
             embedder=str(embedder), created=time.time(), **extra)


def load(path):
    """Return (gallery, cls, col, embedder), or None if there is no usable file.

    `col` is None when the file has no colour data. The embedder name matters because
    vectors from different models cannot be compared.
    """
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return None
    try:
        data = np.load(path)
        col = data['col'] if 'col' in data.files else None
        embedder = str(data['embedder']) if 'embedder' in data.files else ''
        return data['gallery'], int(data['cls']), col, embedder
    except (OSError, ValueError, KeyError):
        return None
