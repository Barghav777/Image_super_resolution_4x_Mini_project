# Image Super Resolution 4x (NTIRE 2025)

This repository contains the code, models, and report for our project on **4x Image Super-Resolution**, submitted as part of the NTIRE 2025 Challenge. Our work focuses on enhancing low-resolution images using an ensemble of state-of-the-art super-resolution models.

## Project Overview

### Introduction

Single Image Super-Resolution (SISR) aims to reconstruct a high-resolution (HR) image from a low-resolution (LR) input. The 4x super-resolution task is challenging due to the severe loss of textures and structural details during downsampling.

In this project, we combined the strengths of three state-of-the-art SISR models (HAT, SwinIR, and RCAN) using an ensemble approach. Our solution consistently achieved superior performance across multiple benchmark datasets.

### Motivation

Super-resolution has critical applications in medical imaging, satellite photography, surveillance, and real-time vision tasks. Our ensemble approach enhances image quality by leveraging the unique strengths of each model.

## Repository Structure

```
├── Final_models
│   ├── hat_model.pth          # Pre-trained HAT model
│   ├── rcan_model.pth         # Pre-trained RCAN model
│   ├── swinir_model.pth       # Pre-trained SwinIR model
│   └── final_ensemble_model.pth # Final ensemble model
├── Other_models               # Models tried but not used in the final ensemble
├── Final_ensemble_train.ipynb  # Training notebook for ensemble
├── Final_test.ipynb           # Testing and evaluation notebook
├── hat_train.ipynb            # Training notebook for HAT model
├── rcan_train.ipynb           # Training notebook for RCAN model
├── swinir_train.ipynb         # Training notebook for SwinIR model
├── Final_report.pdf           # Project report
```

## Models Used

* **HAT (Hybrid Attention Transformer)**

  * Captures both local and global features using channel and spatial attention.
* **SwinIR (Swin Transformer for Image Restoration)**

  * Lightweight and efficient, uses hierarchical feature extraction.
* **RCAN (Residual Channel Attention Network)**

  * Emphasizes informative channels for improved detail preservation.

## Ensemble Strategy

* We use a simple averaging ensemble of the three models:

  ```
  SR_final = (SR_HAT + SR_RCAN + SR_SwinIR) / 3
  ```
* This approach balances the strengths of all models for better image quality.

## Datasets

* Training: LSDIR and Flickr2K (Total: 87,641 images)
* Testing: Benchmark datasets (DIV2K, Set14, Set5, Urban100, BSD100)

## Training Details

* Framework: PyTorch
* Optimizer: Adam with cosine decay scheduling
* Loss Function: L1 Loss
* Batch Size: 32
* Epochs: 200

## Results

### Quantitative Results

* Our ensemble model outperformed individual models in terms of PSNR and SSIM.
* Average Performance:

  * PSNR: 32.02
  * SSIM: 0.772

### Qualitative Results

* The ensemble consistently generated sharper, clearer images compared to individual models.

## How to Use

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```
2. Install the required packages (PyTorch, torchvision, etc.)
3. Run the training notebooks as needed:

   * `hat_train.ipynb` for HAT model
   * `rcan_train.ipynb` for RCAN model
   * `swinir_train.ipynb` for SwinIR model
   * `Final_ensemble_train.ipynb` for ensemble training

## Future Work

* Model Optimization: Speed up inference using pruning and quantization.
* Adaptive Weighting: Dynamically combine model outputs based on confidence.
* Real-Time Use: Deploy the ensemble on edge devices for real-time applications.

## Contributors

* D Barghav (2023UME0253)
* Purushartha Gupta (2023UCE0062)
* Aman Nagar (2023UME0242)
* Misti D Shah (2023UCE0055)

## Supervisor

* Dr. Vinit Jakhetiya, Associate Professor, IIT Jammu
