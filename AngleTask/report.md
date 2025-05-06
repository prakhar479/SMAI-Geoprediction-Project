# Building a Circular CNN Regressor for Image Orientation

We propose using a pretrained CNN backbone (fine-tuned) to regress the 2D orientation angle via a **sine/cosine representation**.  Best practice is to pick a compact but accurate backbone (e.g. ResNet or EfficientNet) and attach a small regression head.  For example, one can feed the 224×224 image through a pretrained CNN (e.g. ResNet-50 or EfficientNet-B0/B3), apply global average pooling, then concatenate any auxiliary inputs (like region ID embedding), and finally output **two values** representing (cosθ, sinθ).  This leverages transfer learning from ImageNet: ResNets are known to achieve state-of-the-art accuracy by training very deep models (e.g. 152 layers with 3.57% ImageNet error), and the EfficientNet family was explicitly designed to maximize accuracy for a given model size.  In particular, EfficientNet-Bx models “transfer well” and achieve SOTA on several vision tasks with an order of magnitude fewer parameters, making them attractive when data are scarce.  Other viable backbones include DenseNet, ResNeXt, or lightweight nets like MobileNet/Vit-small, depending on computational limits.  In all cases we remove the original classification head and add a new dense layer with **2 outputs** (no activation or a tanh to bound them to \[–1,1]).  A typical model architecture might look like:

```
# Pseudocode for model architecture (e.g. in Keras or PyTorch)
image_input = Input(shape=(224,224,3))
features = CNN_Backbone(include_top=False)(image_input)       # e.g. EfficientNet, ResNet
features = GlobalAveragePooling()(features)                   # flatten features
region_input = Input(shape=(1,), dtype=int)                  # region ID integer
region_embed = Embedding(num_regions, embed_dim)(region_input)
region_embed = Flatten()(region_embed)
concat = Concatenate([features, region_embed])               # combine image+region features
x = Dense(units=128, activation='relu')(concat)
x = Dense(units=64, activation='relu')(x)
output = Dense(units=2)(x)  # predicts [cos(theta), sin(theta)]
model = Model([image_input, region_input], output)
```

During training, one can optionally **L2-normalize** the (cos,sin) pair to enforce a unit circle, or rely on the loss to align the vector.  Inference then converts `(cos, sin)` to an angle via `atan2(sin, cos)` (in degrees modulo 360).

## Circular Regression and Loss Functions

Predicting angles requires a *circular* loss.  A natural representation is to encode the target angle θ by its unit-vector \$(\cos\theta,\sin\theta)\$.  The model outputs \$\hat{c}=\cos\hat\theta\$, \$\hat{s}=\sin\hat\theta\$, and we define the loss as the L2 distance on the unit circle:

$$
\mathcal{L} = (\hat{c}-\cos\theta)^2 + (\hat{s}-\sin\theta)^2.
$$

Expanding this, one sees \$\mathcal{L}=2-2\cos(\hat\theta-\theta)\$, which directly penalizes the *angular difference* (small when \$\hat\theta\approx\theta\$).  This chord-length loss correctly handles wrap-around (e.g. 1° vs 359° give small loss).  In practice one can implement it with MSE on the two outputs.  An alternative is to use a cosine-similarity loss or directly minimize \$1-\cos(\hat\theta-\theta)\$.  During evaluation we recover the predicted angle as

```
pred_angle = (atan2(pred_sin, pred_cos) * (180/pi)) mod 360
```

and compute the **Mean Absolute Angular Error** (MAE): for each sample take \$\Delta = \min(|\hat\theta-\theta|, 360-|\hat\theta-\theta|)\$ and average over the test set. This ensures the error is in \[0,180] and reflects the shortest angular difference.

&#x20;notes that using plain MSE on angles fails near the wrap; using the vector loss above avoids that pitfall.

## Incorporating Region ID / Context

Since each image has a *region ID* (fixed location), we can provide this as an auxiliary input.  Treat the ID as a categorical feature: e.g. one-hot encoding or a learned embedding vector.  In practice, a simple approach is to have a parallel branch: the region ID is input to an Embedding layer (or one-hot Dense layer) to produce a feature vector, which is then **concatenated** with the CNN features before the final regression layer.  Tang et al. (2015) use a very similar idea for image classification: they concatenate GPS-derived features (one-hot grid cell, map features, etc.) with CNN features just before the softmax.  They report that naive early concatenation gave no gain, but “concatenating the different feature types before the softmax” (i.e. at a high level) is effective.  In our case, the region ID embedding is analogous to a learned geographic prior.  For example, if there are \$N\$ regions, use an Embedding(\$N,d\$) layer (with \$d\sim8\$–16) and concatenate to the pooled CNN output.  A fully-connected fusion layer then jointly predicts sin/cos.

**Benefits:**  Incorporating region context can significantly improve accuracy.  In Tang et al.’s classification experiments, adding location context yielded nearly a 7% gain in mAP.  By analogy, a region ID embedding should help resolve ambiguities in orientation: e.g. certain landmarks or shadows in a particular location may correlate with direction.

## Data Augmentation Strategies

To mitigate the small data size (\~6500 images), use aggressive image augmentation.  Standard augmentations include:

* **Geometric transforms:** random cropping (with resizing to 224×224), small translations or perspective warps.  Care must be taken with *rotations*: if you rotate the image by \$\alpha\$ degrees, **also add \$\alpha\$ to the label** (mod 360).  Full 360° rotations are possible if labels are adjusted; even a few degrees of random rotation can simulate viewpoint jitter.
* **Horizontal flips:** this is especially useful.  Under a left-right flip, the new angle becomes \$(360-\theta)\bmod 360\$ (e.g. east ↔ west) — account for this label change.  Vertical flips may not apply if the “up” direction is defined (though if the camera tilt changes, the effect on bearing is unclear and often omitted).
* **Color and lighting:** random brightness/contrast, saturation or hue jitter, or Gaussian blur/noise to simulate different times of day or weather.  Since images are taken over \~1 week, include augmentations that cover day/night or sun angle (e.g. add gaussian color distortions or use *ColorJitter*).
* **Occlusion/Noise:** optionally use Cutout or random erasing to make the model robust to occlusions (which do not affect the true bearing).
* **Library examples:** Use augmentation libraries like **Albumentations** or **torchvision.transforms**.  For example, `torchvision.transforms.RandomHorizontalFlip(p=0.5)` (with label fix), `ColorJitter(brightness=0.3, contrast=0.3)`, etc.  Each batch can include a mix of such variants to expand the effective training set.

Be cautious not to create label noise: always apply the same angular shift to the label when transforming.  Avoid heavy augmentations that break the scene (e.g. mixing images, since each image has a single true orientation).  With these augmentations, the model sees a richer distribution of camera conditions, improving generalization.

## Training Procedure and Regularization

**Fine-tuning strategy:** Start by loading the pretrained backbone (with ImageNet weights) and *freeze* most layers initially.  Train the new top layers (embedding+FC layers) for a few epochs with a moderate learning rate (e.g. \$1e^{-3}\$ Adam), to let the newly added weights stabilize.  Then **gradually unfreeze** some of the deeper convolutional layers and continue fine-tuning with a smaller learning rate (e.g. \$1e^{-4}\$ or lower).  This “layer-wise” fine-tuning helps adapt high-level features to our orientation task while preserving generic filters in early layers.

**Regularization:** Use weight decay (L2) on all weights (e.g. \$10^{-4}\$ to \$10^{-5}\$) to prevent overfitting on only 6500 images.  Adding a small Dropout (e.g. 0.2–0.5) in the dense layers can also help.  Batch normalization layers in the backbone should be left in inference mode when frozen, or gently fine-tuned with a low learning rate.  If overfitting remains an issue, consider data augmentation (as above) or early stopping based on validation error.

**Learning rate scheduling:** A cosine annealing or step decay schedule often helps.  For example, use an initial LR warm-up (to avoid large updates at start), then cosine-decay to zero over epochs.  Alternatively, ReduceLROnPlateau on the validation loss or a fixed step drop (e.g. \$\times0.1\$ every few epochs).  Empirically, one-cycle policies (Smith et al.) can yield fast convergence.

**Loss and output constraints:** Since we output (sin, cos), no explicit activation is needed; you can optionally wrap the final layer with `tanh` to restrict $\[-1,1]\$ or manually normalize the output vector:

```python
vec = torch.stack([out_cos, out_sin], dim=1)
vec = vec / vec.norm(dim=1, keepdim=True)  # ensure unit length
```

and then compute loss against the true unit vector.  In practice, even without strict normalization, MSE loss encourages the correct ratio.

**Evaluation metrics:** Always monitor the **mean angular error** on a validation split (or via cross-validation).  Compute it as

```python
err = abs(pred_angle - true_angle)
if err > 180: err = 360 - err
mae = mean(err)
```

in degrees.  Report this MAE, and optionally the Root-Mean-Square Angular Error for continuity.  You may also track the *vector correlation* (cosine similarity averaged) or the regression loss itself.  Because angles are circular, also check e.g. the percentage of predictions within 15° of truth as a secondary metric for practical accuracy.

Finally, validate on unseen region IDs (if possible) to test generalization: you might hold out entire regions during training and test how well the model leverages visual cues alone.  However, in typical use you’ll have region ID for all inputs, so test on a random split and ensure balanced representation.

**Summary:** Use a compact pretrained CNN (ResNet/EfficientNet) with a small custom head for sin/cos output.  Train with a circular loss as above, embed the region ID and merge late in the network, and employ rich augmentations (with label updates) to combat overfitting.  This approach is supported by practice and literature: for instance, residual networks yield state-of-art features and EfficientNets excel at transfer learning, while Tang et al. showed that adding geographic context via concatenation improves vision models. Following these principles should yield a model that predicts valid \[0,360) angles with low mean angular error while remaining sample-efficient and generalizable.

.
