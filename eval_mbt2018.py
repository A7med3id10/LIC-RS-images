import glob, io, math, random
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from compressai.zoo import mbt2018_mean


random.seed(42)
files = sorted(glob.glob("UCMerced_LandUse/*/*.tif"))
files = random.sample(files, 20)   

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * math.log10(255 ** 2 / mse)

dev = "cuda" if torch.cuda.is_available() else "cpu"

mbt_bpp_list, mbt_psnr_list = [], []
j2k_bpp_list, j2k_psnr_list = [], []

for q in range(1, 9):
    net = mbt2018_mean(quality=q, pretrained=True).eval().to(dev)
    net.update(force=True)
    nb, npsnr, jb, jpsnr = [], [], [], []

    for f in files:
        img = Image.open(f).convert("RGB")
        w, h = img.size
        
        
        w_pad = (64 - w % 64) % 64
        h_pad = (64 - h % 64) % 64
        if w_pad > 0 or h_pad > 0:
            img = img.crop((0, 0, w + w_pad, h + h_pad))
            w, h = img.size
            
        ref = np.array(img)
        x = torch.from_numpy(ref).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
        
        with torch.no_grad():
            out = net.compress(x)
            rec = net.decompress(out["strings"], out["shape"])["x_hat"]
            
        bpp = sum(len(s[0]) for s in out["strings"]) * 8 / (w * h)
        rec8 = (rec.clamp(0, 1) * 255).round().byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
        nb.append(bpp)
        npsnr.append(psnr(ref, rec8))

        
        buf = io.BytesIO()
        img.save(buf, format="JPEG2000", quality_mode="rates",
                 quality_layers=[24 / bpp], irreversible=True)
        jb.append(buf.tell() * 8 / (w * h))
        buf.seek(0)
        jpsnr.append(psnr(ref, np.array(Image.open(buf).convert("RGB"))))

    avg_nb, avg_npsnr = np.mean(nb), np.mean(npsnr)
    avg_jb, avg_jpsnr = np.mean(jb), np.mean(jpsnr)

    mbt_bpp_list.append(avg_nb)
    mbt_psnr_list.append(avg_npsnr)
    j2k_bpp_list.append(avg_jb)
    j2k_psnr_list.append(avg_jpsnr)

    print(f"q={q}  mbt2018-mean: bpp={avg_nb:.3f} PSNR={avg_npsnr:.2f} | "
          f"JPEG2000: bpp={avg_jb:.3f} PSNR={avg_jpsnr:.2f}")

# ==========================================
# Rate-Distortion Curve  
# ==========================================
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(mbt_bpp_list, mbt_psnr_list, 'o-', color='#1f77b4', linewidth=2, label='mbt2018-mean')
plt.plot(j2k_bpp_list, j2k_psnr_list, 's--', color='#d62728', linewidth=2, label='JPEG2000')

plt.title('Rate-Distortion Performance on UC Merced Dataset', fontsize=14, fontweight='bold')
plt.xlabel('Bitrate [bpp]', fontsize=12)
plt.ylabel('PSNR [dB]', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11, loc='lower right')
plt.tight_layout()


plt.savefig('rd_curve.png', dpi=300)
plt.show()


# py -3.12 eval_mbt2018.py