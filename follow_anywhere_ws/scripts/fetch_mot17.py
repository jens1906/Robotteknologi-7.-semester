#!/usr/bin/env python3
"""Download MOT17 training sequences (images + ground truth) without the 5.9 GB zip.

The MOTChallenge server supports HTTP range requests, so single sequences are read straight out
of MOT17.zip. Result per sequence: data/mot17_XX/{img1/*.jpg, seqinfo.ini, gt/gt.txt}, which the
detector plays with `-p use_dataset:=true -p video_path:=data/mot17_XX` and scores against gt.txt.

Usage: scripts/fetch_mot17.py [SEQ ...] [--out DIR] [--force]
  SEQ   sequence numbers 02 04 05 09 10 11 13 (default: all seven training sequences)
Only img1/NNNNNN.jpg, seqinfo.ini and gt/gt.txt are written. MOT17 is for non-commercial research.
"""
import argparse
import io
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile

URL = 'https://motchallenge.net/data/MOT17.zip'
TRAIN = ('02', '04', '05', '09', '10', '11', '13')
VARIANT = 'FRCNN'          # images and ground truth are identical in DPM / FRCNN / SDP
ALLOWED = re.compile(r'^(img1/\d{6}\.jpg|seqinfo\.ini|gt/gt\.txt)$')


class HttpFile(io.RawIOBase):
    """Read-only seekable file over HTTP range requests, with read-ahead."""
    AHEAD = 1 << 20

    def __init__(self, url):
        self.url = url
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=30) as r:
            self.size = int(r.headers['Content-Length'])
        self.pos = 0
        self.fetched = 0
        self.buf_start, self.buf = 0, b''

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def _fetch(self, start, end):
        req = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{end - 1}'})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    return r.read()
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))

    def read(self, n=-1):
        if n < 0 or self.pos + n > self.size:
            n = self.size - self.pos
        if n <= 0:
            return b''
        if not (self.buf_start <= self.pos and self.pos + n <= self.buf_start + len(self.buf)):
            self.buf = self._fetch(self.pos, min(self.size, self.pos + max(n, self.AHEAD)))
            self.buf_start = self.pos
            self.fetched += len(self.buf)
        offset = self.pos - self.buf_start
        data = self.buf[offset:offset + n]
        self.pos += len(data)
        return data


def fetch(zf, seq, out_dir, force):
    name = f'MOT17-{seq}-{VARIANT}'
    prefix = f'MOT17/train/{name}/'
    dest = os.path.join(out_dir, f'mot17_{seq}')
    if os.path.isdir(dest) and not force:
        print(f'{dest} exists, skipping (use --force to download again)')
        return
    todo = [i for i in zf.infolist()
            if i.filename.startswith(prefix) and ALLOWED.match(i.filename[len(prefix):])]
    if not todo:
        print(f'{name}: nothing found in the zip', file=sys.stderr)
        return
    size = sum(i.compress_size for i in todo) / 1e6
    print(f'{name}: {len(todo)} files, {size:.0f} MB', flush=True)
    part = dest + '.part'
    shutil.rmtree(part, ignore_errors=True)
    for k, info in enumerate(todo, 1):
        target = os.path.join(part, info.filename[len(prefix):])
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with zf.open(info) as src, open(target, 'wb') as dst:
            dst.write(src.read())
        if k % 100 == 0 or k == len(todo):
            print(f'  {k}/{len(todo)}', flush=True)
    shutil.rmtree(dest, ignore_errors=True)
    os.rename(part, dest)
    print(f'  -> {dest}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('seqs', nargs='*', help='sequence numbers, default all training sequences')
    ap.add_argument('--out', default='data', help='output folder (default data)')
    ap.add_argument('--force', action='store_true', help='download sequences that already exist')
    args = ap.parse_args()
    seqs = [s.replace('MOT17-', '').zfill(2) for s in args.seqs] or list(TRAIN)
    bad = [s for s in seqs if s not in TRAIN]
    if bad:
        ap.error(f'only training sequences have ground truth: {", ".join(TRAIN)} (got {bad})')
    http = HttpFile(URL)
    zf = zipfile.ZipFile(http)
    for seq in seqs:
        fetch(zf, seq, args.out, args.force)
    print(f'done, {http.fetched / 1e6:.0f} MB transferred')


if __name__ == '__main__':
    main()
