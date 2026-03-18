# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Speckle2Self is a PyTorch implementation of "Speckle2Self: Self-Supervised Ultrasound Speckle Reduction Without Clean Data" (Medical Image Analysis, 2025). It trains a multi-resolution encoder-decoder neural network to reduce speckle noise in ultrasound images without requiring clean ground truth data.

## Commands

### Install dependencies
```bash
pip install -r requirements.txt
```

### Training
```bash
python train.py
```
The training script loads config from `configs/params_Simulator.yaml` by default. To use in-vivo data, edit `train.py` to point to `configs/params_inVivo.yaml`.

### Inference
```bash
python inference.py --data_path <input.npy> --model_path <model.pth> --output_path <output.npy> [--visualize]
```

### No test suite or linter is configured in this project.

## Architecture

### Network (`networks/srn/net.py`)
`SpeckleReductionNet` uses **3 parallel encoders** (one per resolution scale: 1x, 0.5x, 0.25x) feeding into a **single shared decoder**. Each encoder has 4 conv blocks + residual blocks, channels: 1→32→64→128→256. The decoder mirrors this with transposed convolutions. A `fuse` flag exists for cross-resolution feature fusion but is currently disabled.

### Training (`utils/training_utils.py`)
- **Reconstruction loss** (L2): compares predictions to input images
- **Consistency loss** (L1): enforces coherence across multi-resolution predictions
- Losses are combined with configurable weights per resolution scale

### Datasets (`utils/datasets.py`)
Two dataset classes:
- `DenoisingDatasetCCA`: Self-supervised — single noisy images as `.npy` arrays of shape `(N, H, W)`. Multi-scale versions created dynamically.
- `DenoisingDatasetSimulator`: Supervised — paired noisy/clean images as `.npy` arrays of shape `(N, 2, H, W)`.

Both use albumentations for augmentation (flips, shift/scale/rotate).

### Image utilities (`utils/image_ops.py`)
Linear normalization to [0,1] and multi-scale resizing with configurable interpolation.

## Data Requirements

- Input must be ultrasound **envelope data** (not B-mode)
- Minimum resolution: 512×512 pixels
- Format: NumPy `.npy` files

## Configuration

YAML configs in `configs/` control hyperparameters: learning rate (default 0.001), batch size (16), epochs (5000), loss weights, optimizer betas, checkpoint/visualization intervals.
