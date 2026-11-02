#!/usr/bin/env python3
"""
JPEG2000 baseline on UC Merced Land Use

Steps
  1. split : build a seeded, class-stratified train/val/test split -> splits/*.csv
  2. eval  : encode/decode the TEST split with JPEG2000 at several target rates,
             measure actual bpp, PSNR, MS-SSIM, MS-SSIM (dB) -> results/*.csv
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


class UCMercedDataset:
    """Minimal dataset: returns (PIL RGB image, class_name, relative_path).
    Works as a torch Dataset too (has __len__/__getitem__)."""
 
    def __init__(self, root, split_csv):
        self.root = Path(root)
        self.items = read_split(Path(split_csv))
        self.classes = sorted({c for _, c in self.items})
 
    def __len__(self):
        return len(self.items)
 
    def __getitem__(self, i):
        rel, c = self.items[i]
        return Image.open(self.root / rel).convert("RGB"), c, rel


# --------------------------------------------------------------------------- #
# JPEG2000 + metrics
# --------------------------------------------------------------------------- #
def j2k_roundtrip(img: Image.Image, target_bpp: float):
    """Encode with OpenJPEG (via Pillow) at a target rate; return (recon, actual_bpp)."""
    ratio = 24.0 / target_bpp  # 24 bits/pixel for 8-bit RGB
    buf = io.BytesIO()
    img.save(
        buf, "JPEG2000",
        quality_mode="rates",
        quality_layers=[ratio],
        irreversible=True,   # 9/7 wavelet (lossy)
        mct=1,               # RGB->YCC colour transform (ignored by old Pillow versions)
        num_resolutions=6,
    )
    data = buf.getvalue()
    rec = Image.open(io.BytesIO(data)).convert("RGB")
    bpp = 8.0 * len(data) / (img.width * img.height)
    return rec, bpp
 
 
def psnr(a: np.ndarray, b: np.ndarray):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
 
 
def make_msssim():
    import torch
    from pytorch_msssim import ms_ssim
 
    def f(a: np.ndarray, b: np.ndarray):
        ta = torch.from_numpy(a).permute(2, 0, 1)[None].float()
        tb = torch.from_numpy(b).permute(2, 0, 1)[None].float()
        return float(ms_ssim(ta, tb, data_range=255.0, size_average=True))
    return f
 
 
def msssim_db(v):
    return -10 * np.log10(max(1 - v, 1e-10))
 
 
def run_eval(args):
    import pandas as pd
    import PIL
 
    ds = UCMercedDataset(args.root, args.split)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    msssim = make_msssim()
    print(f"Test images: {len(ds)} | classes: {len(ds.classes)} | rates: {args.rates}")
 
    rows = []
    for ti, target in enumerate(args.rates):
        for i in range(len(ds)):
            img, cls, rel = ds[i]
            rec, bpp = j2k_roundtrip(img, target)
            a, b = np.asarray(img), np.asarray(rec)
            ms = msssim(a, b)
            rows.append(dict(path=rel, **{"class": cls}, target_bpp=target, bpp=bpp,
                             psnr=psnr(a, b), msssim=ms, msssim_db=msssim_db(ms)))
        print(f"  rate {target:.2f} bpp done ({ti + 1}/{len(args.rates)})")
 
    df = pd.DataFrame(rows)
    df.to_csv(out / "jp2k_per_image.csv", index=False)
 
    agg = (df.groupby("target_bpp")
             .agg(actual_bpp=("bpp", "mean"), psnr_db=("psnr", "mean"),
                  msssim=("msssim", "mean"), msssim_db=("msssim_db", "mean"),
                  n=("path", "count"))
             .reset_index())
    agg.to_csv(out / "jp2k_baseline_table.csv", index=False)
 
    # per-class table (needed for Milestone 3, cheap to produce now)
    per_class = (df.groupby(["class", "target_bpp"])
                   .agg(actual_bpp=("bpp", "mean"), psnr_db=("psnr", "mean"),
                        msssim_db=("msssim_db", "mean"))
                   .reset_index())
    per_class.to_csv(out / "jp2k_per_class.csv", index=False)
 
    with open(out / "env.json", "w") as fh:
        json.dump({"python": platform.python_version(), "pillow": PIL.__version__,
                   "numpy": np.__version__, "rates": args.rates,
                   "split": str(args.split)}, fh, indent=2)
 
    print("\nJPEG2000 baseline (mean over test set)")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved to", out)
 
 



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

    e = sp.add_parser("eval")
    e.add_argument("--root", required=True)
    e.add_argument("--split", default="splits/test.csv")
    e.add_argument("--rates", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    e.add_argument("--out", default="results")

    args = p.parse_args()



    if args.cmd == "split":
        make_split(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
