# Model Architecture

&#x20;We propose a single end-to-end model with a shared image encoder and two output heads (one for latitude, one for longitude).  In practice, **lightweight CNN backbones** such as EfficientNet-B0, MobileNetV3, or a small ResNet (e.g. ResNet‑34) are recommended, as they have proven effective on limited data and can be fine-tuned via transfer learning.  Alternatively, small Vision Transformers (ViT) pretrained on large image corpora can be adapted – for example, one approach uses a pretrained ViT to extract “geo-embeddings” from street images.  The encoder output is then split into two parallel regression branches.  This multi-task design (shared encoder + task‑specific decoders) helps the network learn common spatial features while tuning each head to its specific target.  Each head consists of a few fully-connected layers that regress a continuous value.  (Optionally, one could use a heatmap+DSNT layer per head, since the Differentiable Spatial-to-Numerical Transform preserves spatial generalization in coordinate regression. For simplicity we instead use direct MSE regression.)

Key design points: use global average pooling or flattening after the CNN, then two small MLP heads for lat/lon.  For example, after a backbone with 7×7 spatial output, apply GAP, then concatenate any auxiliary features (below), followed by two separate dense branches.  A regularization such as a small dropout (e.g. 0.2–0.5) may be added in the heads to prevent overfitting.  In summary, a suitable architecture is **Backbone→(Global Pool)→\[Dense→ReLU→Dropout→Output]\_lat** and similarly for longitude.

|       Component      |           Example Models           | Description                                                                                                                              |
| :------------------: | :--------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------- |
|     **Backbone**     |     EfficientNet‑B0, ResNet‑34     | Lightweight CNN pretrained on ImageNet (fine-tuned for regression).  Or use a small ViT (e.g. ViT‑Base/16) pretrained on large datasets. |
| **Regression Heads** | Fully Connected layers (2 outputs) | Two independent branches (multi-task). Each takes the shared image features (plus fused metadata) and outputs a single scalar.           |
| **Coordinate Layer** |   (Optional) DSNT or Soft-argmax   | Alternative to FC: convert a spatial feature map to numeric coordinate, preserving spatial differentiability.                            |
|    **Activation**    |   ReLU (hidden), Linear (output)   | Use ReLU or GELU in hidden layers; linear (no activation) at output to predict continuous values.                                        |

## Input Encoding: RegionID and Angle

Besides the image, we incorporate the **RegionID** (a categorical variable) and **camera angle** (continuous 0–360°).  These are fused with the image features before the final regression.  Best practice is to convert RegionID into a learned embedding vector. In other words, map each RegionID to an index and use an `Embedding(num_regions, embed_dim)` layer.  Learned embeddings allow the model to place similar regions nearby in latent space, avoiding the pitfalls of one-hot encoding.  If the number of regions is small (<10), one-hot plus a linear layer is also acceptable, but for larger or unknown cardinality embeddings are more efficient. The embedding dimension can be modest (e.g. 8–32).

For the **angle**, we exploit its cyclical nature by encoding it as two values: \$\sin(\theta)\$ and \$\cos(\theta)\$ (with \$\theta\$ in radians).  This ensures continuity at the 0/360° wrap and provides a smooth circular encoding (a standard trick in time-series and cyclical features).  No embedding is needed for angle; simply compute these two values (optionally scaled to \[–1,1]) as numerical inputs.

These metadata features are concatenated with the CNN feature vector after pooling.  For example:  after global pooling of the image features, append the RegionID embedding and the two angle features, then feed the combined vector into the regression heads.  (In PyTorch this can be implemented by `torch.cat((img_feats, embed(region_id), sin(theta), cos(theta)), dim=1)`.)  This mixed-input design has been shown to improve performance compared to using the image alone, by providing explicit geographic context.

## Preprocessing & Data Augmentation

**Preprocessing:**  All input images (224×224) should be normalized to zero mean/unit variance (using ImageNet means/std if using a pretrained backbone) and possibly standardized (pixel values in \[0,1]).  The target coordinates are already scaled to \[0,1] and can be used directly.  Auxiliary inputs are also normalized: angle features are in \[–1,1] after sin/cos, and RegionID embeddings are learned.

**Augmentation:**  With only \~6500 samples, aggressive augmentation is key to avoid overfitting. Apply a variety of transformations that preserve geographic cues. Recommended augmentations include:

* **Random Cropping/Resizing:** Randomly crop a region (e.g. scale 0.8–1.0) and resize to 224×224. This simulates slight changes in camera position and zoom, improving robustness to framing. The cited remote-sensing review finds that **random cropping** often yields the largest accuracy gains.
* **Color Jitter:** Randomly adjust brightness, contrast, saturation, and hue within small ranges. This handles varying daylight, weather, or camera settings without altering spatial content.
* **Gaussian Blur or Noise:** Apply slight blur or additive noise to simulate sensor noise or focus changes. Use mild parameters so the scene remains recognizable.
* **Affine/Perspective Transform:** Small random rotations (±15°), horizontal shifts, and mild perspective warps can simulate different viewpoints. (Be cautious: a horizontal flip would invert left-right orientation, so if applied, the angle input must be adjusted: e.g. new\_angle = (360° – old\_angle) mod 360°. Vertical flips are generally not used for street scenes.) Controlled affine or perspective jitter is beneficial.
* **Random Erasing:** Randomly occlude a small patch of the image. This forces the model to use multiple cues and reduces reliance on any single landmark.

The goal is to generate realistic variations. **Avoid augmentations that break semantics:** e.g. do not insert unrelated objects or apply extreme distortions.  Since lat/long depend on visual landmarks and sky/ground features, augmentations should not alter the fundamental geography.  Using **Mixup** or **CutMix** is less common for regression on images, but could be experimented with if overfitting persists.

Empirically, such augmentation techniques can significantly improve generalization on small datasets, reducing model sensitivity to specific image conditions.

## Training Configuration

* **Loss Function:** Use Mean Squared Error (MSE) on each output (latitude and longitude). Since the metric is MSE, the loss \$L = \frac12\[(y\_{\text{lat}} - \hat y\_{\text{lat}})^2 + (y\_{\text{lon}} - \hat y\_{\text{lon}})^2]\$ (or average of the two MSEs) directly aligns with the evaluation. Optionally, one could use **Huber loss (smooth L1)** to reduce sensitivity to outliers, but standard MSE is a suitable default.

* **Optimizer:** Adam (or AdamW for decoupled weight decay) is a good default.  For example, start with learning rate \$\alpha = 10^{-3}\$ or \$10^{-4}\$.  Use a **weight decay** (L2 regularization) on the convolutional weights (e.g. \$10^{-4}\$) to further prevent overfitting.

* **Learning Rate Scheduler:** Use a scheduler such as **ReduceLROnPlateau** (reduce \$\alpha\$ by a factor when validation loss plateaus) or **Cosine Annealing with Warm Restarts**.  A common strategy: train with constant LR for a few epochs, then decay by 10× when the val loss stalls.  Cyclical or cosine schedules can also help converge more robustly.

* **Regularization:** In addition to weight decay, apply **Dropout** in the fully-connected heads (e.g. dropout=0.2–0.5) to discourage reliance on any single neuron.  Batch Normalization in intermediate layers can stabilize training but may not be needed if using a pretrained backbone with its own norms.

* **Batch Size:** Choose based on GPU memory. For 6500 images, a moderate batch (e.g. 16–32) is reasonable.  A smaller batch (16) adds regularization (more gradient noise) but training is noisier; a larger batch (32–64) yields smoother gradients.  Experiment for best trade-off.

* **Epochs & Early Stopping:** Train up to, say, 50–100 epochs, but employ **early stopping** on validation MSE. For example, if the val loss does not improve for 5–10 consecutive epochs, stop training and restore the best model.  This prevents overfitting.  Plotting training vs. validation MSE curves will confirm when the model begins to overfit.

* **Hyperparameter Tuning:** Use a held-out validation set (or cross-validation, see below) to tune learning rate, batch size, dropout rate, etc.  A table of typical values:

  | Hyperparameter    | Example Setting                    | Notes                               |
  | ----------------- | ---------------------------------- | ----------------------------------- |
  | Loss              | MSE (\$L\_2\$)                     | Directly matches metric.            |
  | Optimizer         | Adam (lr=1e-3) + Weight Decay 1e-4 | AdamW is also good.                 |
  | LR Scheduler      | ReduceLRonPlateau (factor=0.1)     | Or Cosine Annealing.                |
  | Dropout           | 0.3 (in regression heads)          | Prevents overfitting on small data. |
  | Batch Size        | 16–32                              | Depends on GPU.                     |
  | Epochs            | 50–100                             | With early stopping.                |
  | Early Stopping    | Patience 5–10                      | Stop if no val improvement.         |
  | Weight Decay (L2) | 1e-4                               | Regularization for conv layers.     |

## Transfer Learning and Pretraining

Given the small dataset size, **transfer learning** is essential.  Initialize the CNN backbone with weights pretrained on a large dataset like ImageNet.  These pretrained feature maps capture general visual patterns (edges, textures, objects) that are useful for geolocation cues. As TensorFlow notes, “if a model is trained on a large and general enough dataset, \[it] will effectively serve as a generic model of the visual world. You can then take advantage of these learned feature maps without having to start from scratch”.  Freeze lower layers initially and fine-tune higher layers and the new heads on your data.

For further improvement, consider **self-supervised pretraining** on unlabeled data. Techniques like SimCLR, MoCo, or BYOL could be applied to a larger collection of street-level images (if available) to learn robust embeddings before supervised fine-tuning.  Recent work shows that self-supervised learning can boost performance on tasks with limited labeled data, although generic methods sometimes struggle to generalize to natural ground images.  If domain-specific pretraining data is available (e.g. street photos from the same regions), it can be used to pretrain the encoder.

Another transfer avenue is **multimodal or multi-task pretraining**. For example, models like CLIP or GeoCLIP (which align images with geographic metadata) can be adapted.  The referenced Stanford project even trains a ViT with contrastive loss to predict geocell clusters. At minimum, using a pretrained ViT or CNN as-is will capture useful priors.

In practice, the simplest approach is to use an ImageNet-pretrained CNN and fine-tune all layers on the latitude/longitude regression. If performance is still lacking, explore domain-specific pretraining or contrastive/self-supervised pretraining on related imagery.

## Evaluation & Validation Strategy

To reliably assess the model and avoid overfitting, use a robust validation protocol:

* **Train/Validation/Test Split:** Reserve a portion of the data as a held-out test set (e.g. 10–20%) that is not used in training or hyperparameter tuning. The remaining data can be split into training and validation (e.g. 80/20 split).

* **Stratification by Region:** Since RegionID encodes location context, ensure that each region is represented in all splits. Use stratified sampling on the RegionID so that the distribution of regions in train/val/test sets is similar. This prevents the model from seeing only one region during training and failing to generalize to others.

* **K-Fold Cross-Validation:** Given the small dataset, perform **K-fold cross-validation** (e.g. K=5 or 10) for more robust estimates of performance. Use group- or stratified-KFold such that images from the same region appear proportionally across folds. For example, one can treat RegionID as a “group” and apply GroupKFold to avoid placing all images of a rare region in a single fold.

* **Metrics:** Report Mean Squared Error (MSE) on latitude and longitude, as specified. Also consider reporting the root MSE or errors in meters (via Haversine formula) for interpretability, though these are auxiliary. Plot predicted vs. actual lat/lon scatter to visualize errors and check for biases. Track both per-axis MSE (lat vs lon) to see if one is learned better.

* **Baseline Comparison:** Compare the model against simple baselines, e.g. predicting the regional centroid (mean lat/lon per RegionID) or a nearest-neighbor retrieval in feature space. This ensures the CNN is learning useful cues beyond trivial strategies.

* **Logging & Monitoring:** During training, log training and validation loss curves (MSE vs epoch) using TensorBoard or a similar tool. This allows early stopping and diagnosis of under/over-fitting. Record final train/test MSE for each fold and report mean ± std.

By carefully splitting data and using cross-validation, we ensure the reported MSE is reliable.  In addition, we can perform **leave-one-region-out** testing as an extreme form of validation: train on all but one region and test on the held-out region to measure generalization to unseen locales.

## Implementation Plan

1. **Data Preparation:**

   * Load images and metadata (RegionID, angle, lat, lon).
   * Preprocess images (resize to 224×224, normalize).
   * Encode RegionID as integer indices; compute angle\_sin = sin(angle*π/180), angle\_cos = cos(angle*π/180).
   * Split into train/val/test (or K folds), stratifying by RegionID.

2. **Model Definition:**

   * In PyTorch or TensorFlow, define the backbone CNN (e.g. `torchvision.models.resnet34(pretrained=True)`). Remove its final classification layer.
   * Add a global pooling (if not already present) to get a fixed-length feature vector (e.g. 512-D).
   * Create an `nn.Embedding(num_regions, embed_dim)` for RegionID.
   * The model’s forward pass should:

     * Compute image features `f_img = backbone(image)`.
     * Lookup region embedding `f_reg = embedding(region_id)`.
     * Concatenate `[f_img; f_reg; angle_sin; angle_cos]` into a single vector.
     * Pass through two separate heads:

       ```
       lat = head_lat(concat_features)  
       lon = head_lon(concat_features)  
       ```
     * Each head can be a small MLP (e.g. `FC( dims → 128 → 1)`).
   * Apply no activation to outputs (linear), yielding 2 scalars in \[0,1].

3. **Training Script:**

   * Define loss = MSELoss applied to both outputs.
   * Set up optimizer (e.g. `Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)`).
   * Use LR scheduler (e.g. `ReduceLROnPlateau` with patience=3).
   * In each epoch, iterate over training batches: apply augmentations (RandomCrop, ColorJitter, etc.), forward pass, compute loss, backpropagate, optimizer step.
   * After each epoch, evaluate on validation set and log val MSE.
   * Implement **early stopping**: if val loss hasn’t improved for `patience` epochs, break and restore best model.
   * Save model checkpoints (best-of-run and last epoch).

4. **Evaluation Script:**

   * After training, run the model on the test set to compute final MSE.
   * Plot predicted vs. actual coordinates for analysis.
   * If performing K-fold, train/evaluate on each fold and aggregate metrics.

5. **Logging & Tracking:**

   * Use **TensorBoard**, **Weights & Biases (wandb)**, or similar to log: learning rate, training/validation loss per epoch, histograms of predictions, etc.
   * Record hyperparameters (backbone type, learning rate, batch size, number of epochs, augmentations used).
   * Version control the code (e.g. Git) and maintain a README with setup instructions.

6. **Post-Training Analysis:**

   * Generate error maps: e.g. plot error magnitude vs. RegionID or vs. angle to identify biases.
   * Check for overfitting by comparing train vs val loss curves.
   * Tune and iterate: if errors are high in certain regions, consider collecting more data or region-specific augmentations.

By following this plan—defining a dual-head CNN, carefully encoding metadata, employing strong augmentation, and rigorously validating—we can build a model that effectively regresses scaled latitude and longitude with low MSE. The shared-backbone multi-task design and use of pretrained networks are key to good performance on limited data, while embedding RegionID and encoding angle preserve crucial spatial context. The combination of these elements should yield a robust geolocation regression system.
