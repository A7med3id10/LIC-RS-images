# 🛰️ Learned Image Compression for Remote-Sensing Images

This project studies whether a learned image codec can compress aerial imagery more efficiently than JPEG2000 while preserving reconstruction quality. The intended use case is remote-sensing systems such as small satellites and UAVs, where storage, power, and downlink bandwidth are limited.

## 🔬 Approach

The baseline comparison uses:

- **Dataset:** [UC Merced Land Use aerial image dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html) (also available from this [Kaggle mirror](https://www.kaggle.com/datasets/nguyenquangnhat2100/ucmerced-landuse-dataset)). The project proposal specifies a reproducible sample selected with random seed 42.
- **Learned codec:** CompressAI's pretrained `mbt2018_mean` Mean-Scale Hyperprior.
- **Classical codec:** JPEG2000, encoded at rates corresponding to the learned codec.
- **Metrics:** Actual bitrate in bits per pixel (bpp) and reconstruction PSNR. The proposal also discusses MS-SSIM, runtime, and memory as additional evaluation measures.

The proposal describes a planned edge-focused extension: replace standard convolutions with depthwise-separable convolutions and add lightweight channel-wise context to the entropy model. These are goals for future implementation; the repository currently contains the baseline evaluation script, not that custom codec.

## 📌 Current scope and caveats

The proposal reports a pilot comparison on 20 images and treats that as the minimum viable baseline result; expanding to 100 images and implementing the lightweight codec modifications are planned work. The evaluation script expects the dataset locally and downloads pretrained CompressAI model weights as needed. Its results are a baseline comparison, not evidence that the planned custom architecture has been implemented or evaluated.

The project proposal notes that the Kaggle-hosted dataset mirror declares CC0, but that declaration may not establish rights for every upstream image. The image files are therefore not included in this repository and should not be redistributed based solely on the mirror's license statement.

## 📚 References

- Minnen, Ballé, and Toderici, [“Joint Autoregressive and Hierarchical Priors for Learned Image Compression”](https://arxiv.org/abs/1809.02736), 2018.
- Bégaint et al., [“CompressAI: a PyTorch library and evaluation platform for learned image compression”](https://arxiv.org/abs/2011.03029), 2020.
- Wang et al., [“Towards an Efficient Remote Sensing Image Compression Network with Visual State Space Model”](https://doi.org/10.3390/rs17030425), 2025.
- Ye et al., [“Map-assisted remote-sensing image compression at extremely low bitrates”](https://doi.org/10.1016/j.isprsjprs.2025.03.005), 2025.
