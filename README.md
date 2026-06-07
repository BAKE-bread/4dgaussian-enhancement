# 4DGS-Enhancement: Multi-modal Prior Guided 4D Gaussian Splatting

This repository contains the partial implementation of the research on **Multi-modal Prior Guided Sparse-view 4D Gaussian Splatting**.

By introducing explicit multi-modal priors (Semantic Masks and Monocular Depth), this framework significantly enhances the robustness, geometry consistency, and rendering quality of 4D Gaussian Splatting, especially under highly ill-posed conditions like monocular or sparse-view dynamic videos.

---

## ✨ Key Enhancements & Features

Compared to baseline 3D/4D Gaussian Splatting, this repository introduces the following core capabilities:

- **Robust Sparse-View Geometry (Scale-invariant Log-Pearson Depth Loss)**  
  Instead of relying solely on photometric loss, we integrate depth priors (e.g., from Depth Anything V2) using a novel Log-Pearson correlation loss. This resolves the scale-ambiguity issue in monocular depth estimation and builds a strict, accurate 3D geometry skeleton, eliminating severe "floater" artifacts.

- **Mask-Gated Density Control**  
  A highly targeted densification mechanism. It utilizes semantic masks (e.g., from SAM 2) combined with local depth errors to explicitly guide the cloning and splitting of Gaussians. Optimization resources are forced to focus on dynamic foreground objects, radically suppressing the infinite growth of noisy floaters in the background.

- **Foreground-Weighted Photometric Optimization**  
  Implements a soft mask-weighted photometric loss to decouple dynamic foregrounds from static backgrounds. It prioritizes the reconstruction of moving objects without aggressively masking out the background, preserving overall scene completeness.

- **Dynamic Annealing Strategy**  
  Balances geometry and texture by enforcing strong depth constraints in the early training stages (to build the skeleton) and smoothly decaying the prior weights in later stages (to recover high-frequency, photorealistic textures).

- **Deformation Network Robustness Shield**  
  Introduces bottom-up gradient defense mechanisms (e.g., hard scale-capping and NaN-gradient filtering) to prevent training crashes and Out-of-Memory (OOM) errors during extreme non-rigid deformations.

---

## ⚙️ Installation

The environment setup is similar to the standard 4D-GS framework.


```
MY Install PyTorch Ver. (adjust CUDA version as needed)
pytorch                   2.4.1           py3.8_cuda12.4_cudnn9_0    pytorch
pytorch-cuda              12.4                 h3fd98bf_7    pytorch
```

---

## 📂 Data Preparation

Your dataset should follow the standard format used in 4D-GS, with additional folders for multi-modal priors. The expected structure is:

```
<dataset_path>/
├── sparse/ or .ply/.npy    # COLMAP SfM output
└── <each_cams>/
    ├── depth_maps/         # Monocular depth maps (e.g., generated via Depth Anything V2)
    ├── images/             # Original extracted frames
    └── masks/              # Semantic masks (e.g., generated via SAM 2)
```

---

## 🚀 Training

It is used in the same way as the standard 4D-GS framework. To train a scene using the enhanced pipeline:

```bash
python train.py -s <path_to_dataset_folder> --port 6017 --expname <path_to_output_folder> --configs <path_to_argument_file>
```

**Key Arguments in `train.py`**:

- `-s` : Path to the source dataset.
- `--expname` : Name of the experiment (outputs will be saved in `output/<expname>`).
- `--configs` : Path to an optional configuration file to override default hyperparameters.

---

## 🎥 Rendering & Evaluation

After training, you can render novel views or evaluate the metrics using the provided scripts:

```bash
# Render dynamic scene (novel view synthesis)
python render_dynamic_offset.py --model_path <path_to_output_folder> --skip_train --configs <path_to_argument_file>

# Render an orbiting camera trajectory (for nerfies dataset)
python render_orbit.py --model_path <path_to_output_folder> --skip_train --configs <path_to_argument_file>

# Evaluate metrics (PSNR, SSIM, LPIPS)
python eval_video.py --model_path <path_to_output_folder>
python metrics.py --model_path <path_to_output_folder>

```

You can also use the original render file in the standard 4D-GS framework.

---

## 📝 Citation

**Important note**: This work builds upon and extends the original 4D Gaussian Splatting frameworks. If you use this codebase, please consider citing their foundational papers:

```bibtex
@inproceedings{Wu_2024_CVPR,
    author    = {Wu, Guanjun and Yi, Taoran and Fang, Jiemin and others},
    title     = {4D Gaussian Splatting for Real-Time Dynamic Scene Rendering},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year      = {2024},
    series    = {CVPR},
}

```

---
