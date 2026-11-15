import os
import time
import argparse
from math import log10

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from compressai.losses import RateDistortionLoss

from para import LightweightMBT2018Mean
from data_set import UCMercedDataset


# ---------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------
class PadOrCropTo:
    """Unify the image to size x size: reflect-pad if smaller, crop if larger."""
    def __init__(self, size=256):
        self.size = size

    def __call__(self, x):  # x: (C, H, W)
        _, h, w = x.shape
        ph = max(self.size - h, 0)
        pw = max(self.size - w, 0)
        if ph or pw:
            x = F.pad(x.unsqueeze(0), (0, pw, 0, ph), mode="reflect").squeeze(0)
        return x[:, :self.size, :self.size]


train_transform = transforms.Compose([
    transforms.ToTensor(),
    PadOrCropTo(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    PadOrCropTo(256),
])

# test: no resize / no crop, the real size is kept
test_transform = transforms.Compose([transforms.ToTensor()])


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    if mse == 0:
        return 100.0
    return 10.0 * log10(1.0 / mse)


def configure_optimizers(model, lr, aux_lr):
    """Main optimizer for the network, separate optimizer for the quantiles (aux loss)."""
    params = [p for n, p in model.named_parameters()
              if p.requires_grad and not n.endswith(".quantiles")]
    aux_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and n.endswith(".quantiles")]
    return optim.Adam(params, lr=lr), optim.Adam(aux_params, lr=aux_lr)


def print_param_breakdown(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total / 1e6:.3f}M")
    for name in ["g_a", "g_s", "h_a", "h_s", "entropy_bottleneck"]:
        n = sum(p.numel() for p in getattr(model, name).parameters())
        print(f"  {name:20s} {n / 1e6:.3f}M")


# ---------------------------------------------------------------
# Train / validation (based on likelihoods, fast)
# ---------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, aux_optimizer, criterion, device):
    model.train()
    tot_loss = tot_bpp = tot_psnr = 0.0

    for d in dataloader:
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out = model(d)
        out_criterion = criterion(out, d)
        out_criterion["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        tot_loss += out_criterion["loss"].item()
        tot_bpp += out_criterion["bpp_loss"].item()
        with torch.no_grad():
            tot_psnr += compute_psnr(d, out["x_hat"].clamp(0, 1))

    n = len(dataloader)
    return tot_loss / n, tot_bpp / n, tot_psnr / n


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    tot_loss = tot_bpp = tot_psnr = 0.0

    for d in dataloader:
        d = d.to(device)
        out = model(d)
        out_criterion = criterion(out, d)

        tot_loss += out_criterion["loss"].item()
        tot_bpp += out_criterion["bpp_loss"].item()
        tot_psnr += compute_psnr(d, out["x_hat"].clamp(0, 1))

    n = len(dataloader)
    return tot_loss / n, tot_bpp / n, tot_psnr / n


# ---------------------------------------------------------------
# Real test evaluation (actual compress / decompress, real bitstream)
# ---------------------------------------------------------------
def pad_to_multiple(x, multiple=64):
    h, w = x.shape[-2:]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph == 0 and pw == 0:
        return x
    return F.pad(x, (0, pw, 0, ph), mode="replicate")


@torch.no_grad()
def test_real(model, csv_path, data_root, device):
    model.eval()
    model.update(force=True)  # build the CDF tables for the arithmetic coder

    dataset = UCMercedDataset(csv_path, data_root, test_transform, sanity_check=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    tot_bpp = tot_psnr = 0.0
    for x in loader:
        x = x.to(device)
        _, _, h, w = x.shape

        x_pad = pad_to_multiple(x, 64)
        out = model.compress(x_pad)
        rec = model.decompress(out["strings"], out["shape"])["x_hat"]
        rec = rec[..., :h, :w].clamp(0, 1)  # remove the padding before measuring

        num_bytes = sum(len(s[0]) for s in out["strings"])
        tot_bpp += num_bytes * 8.0 / (h * w)
        tot_psnr += compute_psnr(x, rec)

    n = len(loader)
    return tot_bpp / n, tot_psnr / n


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./UCMerced_LandUse")
    parser.add_argument("--csv_dir", default="./splits")
    parser.add_argument("--N", type=int, default=80)
    parser.add_argument("--M", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)   # 5 = smoke test only
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lmbda", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--aux_lr", type=float, default=1e-3)
    parser.add_argument("--ckpt", default="./checkpoints/best_lightweight_mbt2018.pth.tar")
    parser.add_argument("--eval_only", action="store_true",
                        help="skip training, load the checkpoint and run the real test eval")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Pipeline on {device} ===")

    model = LightweightMBT2018Mean(N=args.N, M=args.M).to(device)
    print_param_breakdown(model)

    if not args.eval_only:
        train_dataset = UCMercedDataset(os.path.join(args.csv_dir, "train.csv"),
                                        args.data_root, train_transform, sanity_check=False)
        val_dataset = UCMercedDataset(os.path.join(args.csv_dir, "val.csv"),
                                      args.data_root, val_transform, sanity_check=False)

        pin = device.type == "cuda"
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=2, pin_memory=pin)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=2, pin_memory=pin)
        print(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")

        criterion = RateDistortionLoss(lmbda=args.lmbda, metric="mse")
        optimizer, aux_optimizer = configure_optimizers(model, args.lr, args.aux_lr)

        os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
        best_loss = float("inf")

        print("\nStarting training loop...")
        for epoch in range(args.epochs):
            t0 = time.time()
            tr_loss, tr_bpp, tr_psnr = train_one_epoch(
                model, train_loader, optimizer, aux_optimizer, criterion, device)
            va_loss, va_bpp, va_psnr = evaluate(model, val_loader, criterion, device)

            print(f"Epoch [{epoch + 1}/{args.epochs}] ({time.time() - t0:.1f}s) | "
                  f"Train Loss {tr_loss:.4f} Bpp {tr_bpp:.4f} PSNR {tr_psnr:.2f}dB || "
                  f"Val Loss {va_loss:.4f} Bpp {va_bpp:.4f} PSNR {va_psnr:.2f}dB")

            if va_loss < best_loss:
                best_loss = va_loss
                torch.save({
                    "epoch": epoch + 1,
                    "N": args.N,
                    "M": args.M,
                    "lmbda": args.lmbda,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "val_loss": va_loss,
                }, args.ckpt)
                print(f"--> Saved best checkpoint to {args.ckpt}")

    # ---- Real test evaluation using the best checkpoint ----
    print("\n=== Real test evaluation (compress / decompress) ===")
    ckpt = torch.load(args.ckpt, map_location=device)
    model = LightweightMBT2018Mean(N=ckpt["N"], M=ckpt["M"]).to(device)
    model.load_state_dict(ckpt["state_dict"])

    bpp, psnr = test_real(model, os.path.join(args.csv_dir, "test.csv"),
                          args.data_root, device)
    print(f"Test (210 images) | real bpp: {bpp:.4f} | PSNR: {psnr:.2f} dB")


if __name__ == "__main__":
    main()

# py -3.12 train_eval.py
# py -3.12 train_eval.py --epochs 100
# py -3.12 train_eval.py --eval_only