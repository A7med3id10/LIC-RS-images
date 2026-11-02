#!/usr/bin/env python3
"""
JPEG2000 baseline on UC Merced Land Use (Student A, Milestone 1).

Steps
  1. split : build a seeded, class-stratified train/val/test split -> splits/*.csv
  2. eval  : encode/decode the TEST split with JPEG2000 at several target rates,
             measure actual bpp, PSNR, MS-SSIM, MS-SSIM (dB) -> results/*.csv

Usage
  python eval_jp2k.py download --root data
  python eval_jp2k.py split --root data/UCMerced_LandUse/Images --out splits --seed 42
  python eval_jp2k.py eval  --root data/UCMerced_LandUse/Images --split splits/test.csv \
                            --rates 0.25 0.5 0.75 1.0 1.5 2.0 --out results

Dependencies (pin these in requirements.txt):
  pip install pillow numpy torch pytorch-msssim pandas
  Pillow must be built with OpenJPEG (the official wheels are).

Metric definitions (keep identical for Students B and C):
  bpp          = 8 * compressed_bytes / (H * W)       (real file size, headers included)
  PSNR         = 10 log10(255^2 / MSE) on 8-bit RGB, computed per image, then averaged
  MS-SSIM      = pytorch_msssim, data_range=255, 5 scales, 11x11 window, per image
  MS-SSIM (dB) = -10 log10(1 - MS-SSIM), computed per image, then averaged
                 (same transform as the VMIC paper, Sec. 4.1.2)
"""
import argparse
import csv
import io
import json
import platform
import random
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
EXPECTED_SIZE = (256, 256)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def list_images(root: Path):
    """Return sorted [(relative_path, class_name)] for root/<class>/<img>."""
    items = []
    for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(cdir.iterdir()):
            if f.suffix.lower() in EXTS:
                items.append((f"{cdir.name}/{f.name}", cdir.name))
    if not items:
        sys.exit(f"No images found under {root}. Expected root/<class>/<image>.tif")
    return items


def sanity_check(root: Path, items):
    bad_count = 0
    for rel, _ in items:
        try:
            with Image.open(root / rel) as im:
                im.load()
                if im.size != EXPECTED_SIZE or im.convert("RGB").mode != "RGB":
                    bad_count += 1
        except Exception as e:
            sys.exit(f"[sanity ERROR] Corrupt file {rel}: {e}")
    print(f"[sanity] Found {bad_count} images requiring auto-resize to {EXPECTED_SIZE}. Proceeding...")


def make_split(args):
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = list_images(root)
    sanity_check(root, items)

    by_class = {}
    for rel, c in items:
        by_class.setdefault(c, []).append(rel)

    rng = random.Random(args.seed)
    split = {"train": [], "val": [], "test": []}
    for c, files in sorted(by_class.items()):
        files = sorted(files)
        rng.shuffle(files)
        n = len(files)
        n_test = max(1, round(n * args.test_frac))
        n_val = max(1, round(n * args.val_frac))
        split["test"] += [(f, c) for f in files[:n_test]]
        split["val"] += [(f, c) for f in files[n_test:n_test + n_val]]
        split["train"] += [(f, c) for f in files[n_test + n_val:]]

    for name, rows in split.items():
        with open(out / f"{name}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["path", "class"])
            w.writerows(sorted(rows))
        print(f"{name}: {len(rows)} images")
    with open(out / "split_info.json", "w") as fh:
        json.dump({"seed": args.seed, "val_frac": args.val_frac,
                   "test_frac": args.test_frac, "n_classes": len(by_class)}, fh, indent=2)
    print("Wrote split files to", out, "(commit these to the repo)")


def read_split(path: Path):
    with open(path) as fh:
        return [(r["path"], r["class"]) for r in csv.DictReader(fh)]



# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("split")
    s.add_argument("--root", required=True, help="folder with <class>/<image> subfolders")
    s.add_argument("--out", default="splits")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--val-frac", type=float, default=0.10)
    s.add_argument("--test-frac", type=float, default=0.10)

    args = p.parse_args()

    if args.cmd == "split":
        make_split(args)


if __name__ == "__main__":
    main()
