# GAMFlow

Code package related to **GAMFlow: Global Attention-Based Flow Model for Anomaly Detection and Localization**.

## Paper

- Title: GAMFlow: Global Attention-Based Flow Model for Anomaly Detection and Localization
- Authors: Fan Zhang, Ruiqing Yan, Jinfeng Li, Jiasheng He, Chun Fang
- Journal: IEEE Access, 2023
- DOI: https://doi.org/10.1109/ACCESS.2023.3326753

## Overview

GAMFlow is an unsupervised industrial anomaly detection and localization model based on feature extraction and flow-based distribution estimation. This repository is organized around a FastFlow-style normalizing-flow backbone, with related attention modules retained for experiments:

- `gamflow/attention/gam.py`: Global Attention Mechanism module.
- `gamflow/attention/cbam.py`: Convolutional Block Attention Module.
- `gamflow/attention/simam.py`: SimAM attention module currently wired into `gamflow/fastflow.py`.

## Repository Contents

- `scripts/train.py`: main training and evaluation entry point.
- `gamflow/`: model, dataset, constants, utilities, and attention modules.
- `configs/`: backbone and flow configuration files.
- `third_party/FrEIA/`: vendored FrEIA implementation used for invertible flow blocks.
- `requirements.txt`: Python dependency list.

## Installation

```bash
pip install -r requirements.txt
```

## Data

This project uses [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad). The dataset should be arranged as:

```text
mvtec-ad/
  bottle/
    train/
    test/
    ground_truth/
  cable/
    train/
    test/
    ground_truth/
  ...
```

## Usage

Train a category:

```bash
python scripts/train.py -cfg configs/cait.yaml --data /path/to/mvtec-ad -cat bottle
```

Evaluate a checkpoint:

```bash
python scripts/train.py \
  --cfg configs/resnet18.yaml \
  --data /path/to/mvtec-ad \
  -cat bottle \
  --eval \
  -ckpt /path/to/checkpoint.pt
```

The script reports image-level and pixel-level AUROC during evaluation.

## Notes

- The original working directory included `main2.py` and `main90090.py`; those were removed because they were duplicate/debug variants of `scripts/train.py`.
- Large datasets and checkpoints are not committed.
