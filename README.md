## Overview

This repository provides an end-to-end solution for predicting geographic information from street‐level imagery. Given a labeled set of 224×224 images, the model jointly estimates:

* **Region ID** (categorical classification into one of 15 regions)
* **Latitude & Longitude** (continuous regression, scaled to \[0,1])
* **Camera Orientation Angle** (continuous regression in \[0,360°])

We employ a shared backbone encoder with task‐specific heads for multi-task learning, fuse auxiliary metadata (region ID, angle encoding) into the regression heads, leverage transfer learning, and apply strong data augmentations to combat overfitting on a small dataset (\~6,500 images).

Note: We also implement a KNN approach for refining our Lat-Long regression task result for better predictions

---

## Model Architecture

### Shared Image Encoder

* **Backbone**: Lightweight CNNs (e.g., EfficientNet-B0, MobileNetV3-Large, ResNet-34) or small ViTs pretrained on ImageNet.
* **Feature Extraction**: Remove original classification head, apply global average pooling (GAP) to obtain a fixed-length feature vector.

### Task Heads

1. **Region Classification Head**

   * MLP: `[Dropout(0.2) → Dense(128) → ReLU → Dropout(0.2) → Dense(15)]`
   * Loss: Cross-Entropy

2. **Latitude & Longitude Regression Heads**

   * Two parallel MLPs: each `[Dense(128) → ReLU → Dropout(0.3) → Dense(1)]`
   * Loss: Mean Squared Error (MSE) per output

3. **Orientation Regression Head**

   * Single MLP: `[Dense(128) → ReLU → Dropout(0.3) → Dense(2)]` predicting `(cos θ, sin θ)`
   * Loss: MSE over unit‐vector representation to handle circularity
   * Angle Recovery: `atan2(sin, cos) → degrees mod 360`

---

## Metadata Fusion

* **Region ID**: Learnable embedding (`Embedding(num_regions=15, dim=16)`), concatenated with image features.
* **Camera Angle**: Encode raw angle θ (degrees) as `[sin(θ·π/180), cos(θ·π/180)]` and concatenate.

After GAP, fuse as: `features = cat([img_feats, region_embed, angle_sin, angle_cos], dim=1)` before feeding into regression heads.

---

## Data Preprocessing & Augmentation

1. **Normalization**

   * Resize images to 224×224
   * Normalize with ImageNet mean/std: `mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]`
2. **Geometric Augmentations**

   * RandomResizedCrop (scale 0.8–1.0)
   * RandomRotation (±15°) — update label by +angle
   * RandomHorizontalFlip — update orientation label: `θ_new = (360 − θ) mod 360`
   * Optional small perspective transforms
3. **Color & Noise**

   * ColorJitter (brightness, contrast, saturation, hue)
   * Gaussian blur or additive noise
4. **Occlusion**

   * Random Erasing / Cutout to improve robustness

Ensure label consistency for augmentations that affect angle.

---

## Training Configuration

* **Optimizers**

  * Stage 1 (Head-only): AdamW on new heads, lr=1e-3, weight\_decay=1e-4, freeze backbone
  * Stage 2 (Fine-tuning): Unfreeze last blocks or all layers, lr=1e-4 or lower
* **Schedulers**

  * CosineAnnealingLR or OneCycleLR
  * ReduceLROnPlateau on validation loss
* **Regularization**

  * Dropout in heads (0.2–0.5)
  * Weight decay (1e-4)
  * Early stopping with patience 5–10 epochs
* **Batch Size & Epochs**

  * Batch size: 16–64 (GPU‐dependent)
  * Epochs: up to 50–100 with early stopping

---

## Validation & Evaluation

* **Data Split**: Stratified train/val/test (e.g., 80/10/10) by Region ID

* **Cross-Validation**: 5‑fold Stratified or GroupKFold on regions

* **Metrics**

  * Classification: Accuracy, Precision/Recall, Confusion Matrix
  * Regression (lat/lon): MSE, RMSE, Haversine distance to estimate meters
  * Angular: Mean Absolute Angular Error (°), % within 15°

* **Baselines**: Regional centroid predictor, nearest‐neighbor in feature space

* **Visualization**: Predicted vs. actual scatter plots, error-vs-region/angle analysis

---

## Implementation Plan

1. **Data Pipeline**: Load images + metadata, apply preprocessing/transforms, create PyTorch `Dataset` & `DataLoader`
2. **Model Definition**: Compose backbone, embedding layer, fusion logic, and task heads in PyTorch/TensorFlow
3. **Training Scripts**: Training loop with mixed precision (optional), logging (TensorBoard/W\&B), checkpointing
4. **Evaluation Scripts**: Compute metrics, produce plots, aggregate cross‐validation results
5. **Hyperparameter Tuning**: Grid-search or Bayesian methods (learning rate, dropout, embedding dim)

![github](https://github.com/prakhar479/SMAI-Geoprediction-Project)
![models](https://1drv.ms/f/c/f88f6002b41bfe05/Ej2OPIgxq-VFjWhWXJxzpaoBmMvubU_mfhs1QdqDQgCeng?e=CKR5lN)