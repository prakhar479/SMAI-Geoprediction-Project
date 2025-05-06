"""
Image Orientation Prediction with Circular CNN Regressor
--------------------------------------------------------
This script trains a CNN model to predict the orientation angle of images.
The model uses pretrained backbones and outputs sine/cosine representation
for circular regression.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import pi
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.transforms.functional as TF

from PIL import Image
from sklearn.model_selection import train_test_split

# Set seeds for reproducibility


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Dataset class for angle prediction


class AngleDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, train=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            train (bool): Whether this is a training set (for augmentation decisions)
        """
        self.angle_data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.train = train

        # Get unique region IDs and create a mapping
        self.unique_regions = self.angle_data['Region_ID'].unique()
        self.region_to_idx = {region: idx for idx,
                              region in enumerate(self.unique_regions)}

    def __len__(self):
        return len(self.angle_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.angle_data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        angle = self.angle_data.iloc[idx, 2]  # Original angle in degrees
        region_id = self.angle_data.iloc[idx, 1]
        region_idx = self.region_to_idx[region_id]

        # Convert angle to radians for sin/cos calculation
        angle_rad = angle * (pi / 180.0)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        target = torch.tensor([cos_angle, sin_angle], dtype=torch.float)

        return image, torch.tensor(region_idx, dtype=torch.long), target

# Custom augmentation class with angle adjustment


class RandomRotationWithAngle:
    def __init__(self, degrees=15):
        self.degrees = degrees

    def __call__(self, img, angle_vector):
        """
        Args:
            img: PIL Image to be rotated
            angle_vector: [cos(theta), sin(theta)]

        Returns:
            rotated image, updated angle vector
        """
        # Sample rotation angle
        rotation_angle = random.uniform(-self.degrees, self.degrees)

        # Rotate image
        rotated_img = TF.rotate(img, rotation_angle)

        # Update angle (convert from vector, add rotation, convert back)
        cos_theta, sin_theta = angle_vector
        original_angle = np.arctan2(sin_theta, cos_theta) * (180.0 / pi)
        new_angle = (original_angle + rotation_angle) % 360
        new_angle_rad = new_angle * (pi / 180.0)

        new_cos = np.cos(new_angle_rad)
        new_sin = np.sin(new_angle_rad)

        return rotated_img, torch.tensor([new_cos, new_sin], dtype=torch.float)


class RandomHorizontalFlipWithAngle:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, angle_vector):
        """Flip the image horizontally with probability p and adjust angle."""
        if random.random() < self.p:
            # Flip image
            flipped_img = TF.hflip(img)

            # Update angle: for horizontal flip, θ becomes (360-θ) % 360
            cos_theta, sin_theta = angle_vector
            original_angle = np.arctan2(sin_theta, cos_theta) * (180.0 / pi)
            new_angle = (360 - original_angle) % 360
            new_angle_rad = new_angle * (pi / 180.0)

            new_cos = np.cos(new_angle_rad)
            new_sin = np.sin(new_angle_rad)

            return flipped_img, torch.tensor([new_cos, new_sin], dtype=torch.float)

        return img, angle_vector

# CNN Model with pretrained backbone


class AnglePredictor(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0', num_regions=1, embed_dim=16, pretrained=True):
        super(AnglePredictor, self).__init__()

        # Select backbone architecture
        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(
                weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = self.backbone.fc.in_features
            self.backbone = nn.Sequential(
                *list(self.backbone.children())[:-2])  # Remove avg pool and fc
        elif backbone_name == 'resnet50':
            self.backbone = models.resnet50(
                weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = self.backbone.fc.in_features
            self.backbone = nn.Sequential(
                *list(self.backbone.children())[:-2])  # Remove avg pool and fc
        elif backbone_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(
                weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone = self.backbone.features  # Keep only the feature extractor
        elif backbone_name == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(
                weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone = self.backbone.features  # Keep only the feature extractor
        elif backbone_name == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(
                weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone = self.backbone.features  # Keep only the feature extractor
        elif backbone_name == 'mobilenet_v3_large':
            self.backbone = models.mobilenet_v3_large(
                weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = self.backbone.features[-1].out_channels
            self.backbone = self.backbone.features
        elif backbone_name == 'convnext_small':
            self.backbone = models.convnext_small(
                weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = self.backbone.classifier[2].in_features
            self.backbone = self.backbone.features
        elif backbone_name == 'convnext_base':
            self.backbone = models.convnext_base(
                weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = self.backbone.classifier[2].in_features
            self.backbone = self.backbone.features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Region ID embedding
        self.region_embedding = nn.Embedding(num_regions, embed_dim)

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(feature_dim + embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)  # Output cos_theta, sin_theta
        )

    def forward(self, x, region_id):
        # Extract features from the backbone
        features = self.backbone(x)
        features = self.global_pool(features)
        features = torch.flatten(features, 1)

        # Get region embeddings
        region_embed = self.region_embedding(region_id)

        # Concatenate features
        concat_features = torch.cat([features, region_embed], dim=1)

        # Predict sin/cos values
        output = self.regression_head(concat_features)

        # Optional: normalize to ensure unit vector (uncomment if needed)
        # output = output / torch.norm(output, dim=1, keepdim=True)

        return output

# Circular loss function (MSE on sin/cos representation)


def circular_loss(pred, target):
    return nn.MSELoss()(pred, target)

# Angular error computation


def compute_angular_error(pred, target):
    """
    Compute the angular error in degrees between prediction and target.

    Args:
        pred: tensor of shape [batch_size, 2] containing [cos, sin] predictions
        target: tensor of shape [batch_size, 2] containing [cos, sin] targets

    Returns:
        Mean Absolute Angular Error in degrees
    """
    # Convert [cos, sin] to angles in degrees
    pred_angle = torch.atan2(pred[:, 1], pred[:, 0]) * (180 / pi)
    target_angle = torch.atan2(target[:, 1], target[:, 0]) * (180 / pi)

    # Convert to [0, 360) range
    pred_angle = (pred_angle + 360) % 360
    target_angle = (target_angle + 360) % 360

    # Compute minimum angle difference (considering the circular nature)
    diff = torch.abs(pred_angle - target_angle)
    diff = torch.min(diff, 360 - diff)

    return diff.mean().item()

# Training function


def train_model(model, train_loader, val_loader, args):
    """
    Train the model and return the trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=3, factor=0.5)

    # Track best model
    best_val_error = float('inf')
    best_model_state = None

    train_losses = []
    val_errors = []

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        for images, region_ids, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images = images.to(device)
            region_ids = region_ids.to(device)
            targets = targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images, region_ids)

            # Calculate loss
            loss = circular_loss(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation phase
        val_error = validate_model(model, val_loader, device)
        val_errors.append(val_error)

        print(
            f"Epoch {epoch+1}/{args.epochs} - Loss: {epoch_loss:.4f} - Val Error: {val_error:.2f}°")

        # Learning rate scheduling
        scheduler.step(val_error)

        # Save best model
        if val_error < best_val_error:
            best_val_error = val_error
            best_model_state = model.state_dict().copy()
            print(
                f"New best model with validation error: {best_val_error:.2f}°")

    # Plot training progress
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_errors)
    plt.xlabel('Epoch')
    plt.ylabel('Angular Error (degrees)')
    plt.title('Validation Error')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"training_progress_{args.backbone}.png"))
    plt.close()

    # Load best model state
    model.load_state_dict(best_model_state)

    return model, best_val_error

# Validation function


def validate_model(model, val_loader, device):
    """
    Validate the model and return the mean angular error.
    """
    model.eval()
    angular_errors = []

    with torch.no_grad():
        for images, region_ids, targets in val_loader:
            images = images.to(device)
            region_ids = region_ids.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(images, region_ids)

            # Calculate angular error
            batch_error = compute_angular_error(outputs, targets)
            angular_errors.append(batch_error)

    return np.mean(angular_errors)

# Function to visualize predictions


def visualize_predictions(model, val_loader, device, num_samples=5):
    """
    Visualize some example predictions vs ground truth
    """
    model.eval()
    images, region_ids, targets = next(iter(val_loader))

    plt.figure(figsize=(15, 3 * num_samples))

    with torch.no_grad():
        images_sample = images[:num_samples].to(device)
        region_ids_sample = region_ids[:num_samples].to(device)
        targets_sample = targets[:num_samples]

        outputs = model(images_sample, region_ids_sample)

        for i in range(num_samples):
            img = images_sample[i].cpu().permute(1, 2, 0).numpy()
            img = np.clip(
                img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)

            true_angle = np.degrees(np.arctan2(
                targets_sample[i, 1].item(), targets_sample[i, 0].item())) % 360
            pred_angle = np.degrees(np.arctan2(
                outputs[i, 1].cpu().item(), outputs[i, 0].cpu().item())) % 360
            error = min(abs(true_angle - pred_angle),
                        360 - abs(true_angle - pred_angle))

            plt.subplot(num_samples, 1, i+1)
            plt.imshow(img)
            plt.title(
                f"True: {true_angle:.1f}° | Pred: {pred_angle:.1f}° | Error: {error:.1f}°")
            plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"predictions_{args.backbone}.png"))
    plt.close()


def main(args):
    # Define data transformations
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Basic transformation for validation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # Augmented transformation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomGrayscale(p=0.02),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        normalize
    ])

    # Create datasets
    train_dataset = AngleDataset(
        csv_file=os.path.join(args.data_dir, 'train_label.csv'),
        img_dir=os.path.join(args.data_dir, 'images_train'),
        transform=train_transform,
        train=True
    )

    val_dataset = AngleDataset(
        csv_file=os.path.join(args.data_dir, 'val_label.csv'),
        img_dir=os.path.join(args.data_dir, 'images_val'),
        transform=val_transform,
        train=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Model setup
    num_regions = len(train_dataset.unique_regions)
    print(f"Number of unique regions: {num_regions}")

    model = AnglePredictor(
        backbone_name=args.backbone,
        num_regions=num_regions,
        embed_dim=args.embed_dim,
        pretrained=args.pretrained
    )

    print(f"Training with {args.backbone} backbone")

    # Train the model
    model, best_val_error = train_model(model, train_loader, val_loader, args)

    # Evaluate on full validation set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_val_error = validate_model(model, val_loader, device)
    print(f"Best validation circular error: {best_val_error:.2f}")
    print(f"Final validation angular error: {final_val_error:.2f}°")

    # Visualize some predictions
    visualize_predictions(model, val_loader, device)

    # Save model
    torch.save(model.state_dict(), os.path.join(
        args.output_dir, f"{args.backbone}_orientation_model.pth"))
    print(f"Model saved to {os.path.join(args.output_dir, f'{args.backbone}_orientation_model.pth')}")

    return final_val_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a circular CNN regressor for image orientation prediction")

    # Data parameters
    parser.add_argument("--data_dir", type=str, default="AngleDataset",
                        help="Directory containing the dataset")

    # Model parameters
    parser.add_argument("--backbone", type=str, default="efficientnet_b0",
                        choices=["resnet18", "resnet50", "efficientnet_b0", "efficientnet_b3",
                                 "mobilenet_v2", "mobilenet_v3_large", "convnext_small", "convnext_base"],
                        help="Backbone CNN architecture to use")
    parser.add_argument("--embed_dim", type=int, default=16,
                        help="Dimension of region embedding")
    parser.add_argument("--pretrained", action="store_true", default=False,
                        help="Use pre-trained backbone weights")

    # Training parameters``
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay (L2 penalty)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker threads for data loading")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save model and results")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    set_seed(args.seed)
    
    final_error = main(args)
    print(f"\nTraining completed.")
    print(f"Model: {args.backbone}")
