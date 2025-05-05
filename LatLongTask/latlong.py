import os
import math
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from PIL import Image, ImageFile

# Ensure PIL can load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class BackboneFactory:
    """Factory class to create and configure different backbone architectures."""
    
    @staticmethod
    def get_backbone(backbone_name, pretrained=True):
        """
        Get a backbone model and its feature dimension.
        
        Args:
            backbone_name (str): Name of the backbone architecture
            pretrained (bool): Whether to use pretrained weights
            
        Returns:
            backbone (nn.Module): The backbone model
            feature_dim (int): Feature dimension for the backbone
        """
        # ResNet family
        if backbone_name == 'resnet18':
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            feature_dim = backbone.fc.in_features
            backbone = nn.Sequential(*list(backbone.children())[:-1])
        elif backbone_name == 'resnet34':
            backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
            feature_dim = backbone.fc.in_features
            backbone = nn.Sequential(*list(backbone.children())[:-1])
        elif backbone_name == 'resnet50':
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            feature_dim = backbone.fc.in_features
            backbone = nn.Sequential(*list(backbone.children())[:-1])
        elif backbone_name == 'resnet101':
            backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT if pretrained else None)
            feature_dim = backbone.fc.in_features
            backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        # EfficientNet family
        elif backbone_name == 'efficientnet_b0':
            backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            feature_dim = backbone.classifier[1].in_features
            backbone = nn.Sequential(*list(backbone.children())[:-1])
        elif backbone_name == 'efficientnet_b1':
            backbone = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT if pretrained else None)
            feature_dim = backbone.classifier[1].in_features
            backbone = nn.Sequential(*list(backbone.children())[:-1])
        elif backbone_name == 'efficientnet_b2':
            backbone = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT if pretrained else None)
            feature_dim = backbone.classifier[1].in_features
            backbone = nn.Sequential(*list(backbone.children())[:-1])
        elif backbone_name == 'efficientnet_b3':
            backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT if pretrained else None)
            feature_dim = backbone.classifier[1].in_features
            backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        # MobileNet family
        elif backbone_name == 'mobilenet_v3_small':
            backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
            feature_dim = backbone.classifier[0].in_features
            backbone = nn.Sequential(*list(backbone.children())[:-1])
        elif backbone_name == 'mobilenet_v3_large':
            backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
            feature_dim = backbone.classifier[0].in_features
            backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        # Vision Transformer family
        elif backbone_name == 'vit_b_16':
            backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)
            feature_dim = backbone.heads.head.in_features
            backbone.heads = nn.Identity()
        elif backbone_name == 'vit_b_32':
            backbone = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT if pretrained else None)
            feature_dim = backbone.heads.head.in_features
            backbone.heads = nn.Identity()
        elif backbone_name == 'vit_l_16':
            backbone = models.vit_l_16(weights=models.ViT_L_16_Weights.DEFAULT if pretrained else None)
            feature_dim = backbone.heads.head.in_features
            backbone.heads = nn.Identity()
        elif backbone_name == 'vit_l_32':
            backbone = models.vit_l_32(weights=models.ViT_L_32_Weights.DEFAULT if pretrained else None)
            feature_dim = backbone.heads.head.in_features
            backbone.heads = nn.Identity()
        
        # Swin Transformer family
        elif backbone_name == 'swin_t':
            backbone = models.swin_t(weights=models.Swin_T_Weights.DEFAULT if pretrained else None)
            feature_dim = backbone.head.in_features
            backbone.head = nn.Identity()
        elif backbone_name == 'swin_s':
            backbone = models.swin_s(weights=models.Swin_S_Weights.DEFAULT if pretrained else None)
            feature_dim = backbone.head.in_features
            backbone.head = nn.Identity()
        elif backbone_name == 'swin_b':
            backbone = models.swin_b(weights=models.Swin_B_Weights.DEFAULT if pretrained else None)
            feature_dim = backbone.head.in_features
            backbone.head = nn.Identity()
        
        # ConvNext family
        elif backbone_name == 'convnext_tiny':
            backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
            feature_dim = backbone.classifier[2].in_features
            backbone.classifier = nn.Identity()
        elif backbone_name == 'convnext_small':
            backbone = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None)
            feature_dim = backbone.classifier[2].in_features
            backbone.classifier = nn.Identity()
        elif backbone_name == 'convnext_base':
            backbone = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None)
            feature_dim = backbone.classifier[2].in_features
            backbone.classifier = nn.Identity()
        
        # RegNet family
        elif backbone_name == 'regnet_y_400mf':
            backbone = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.DEFAULT if pretrained else None)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif backbone_name == 'regnet_y_800mf':
            backbone = models.regnet_y_800mf(weights=models.RegNet_Y_800MF_Weights.DEFAULT if pretrained else None)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif backbone_name == 'regnet_y_1_6gf':
            backbone = models.regnet_y_1_6gf(weights=models.RegNet_Y_1_6GF_Weights.DEFAULT if pretrained else None)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        
        # DenseNet family
        elif backbone_name == 'densenet121':
            backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
            feature_dim = backbone.classifier.in_features
            backbone.classifier = nn.Identity()
        elif backbone_name == 'densenet169':
            backbone = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT if pretrained else None)
            feature_dim = backbone.classifier.in_features
            backbone.classifier = nn.Identity()
        
        # MaxViT family
        elif backbone_name == 'maxvit_t':
            backbone = models.maxvit_t(weights=models.MaxVit_T_Weights.DEFAULT if pretrained else None)
            feature_dim = backbone.classifier[5].in_features
            backbone.classifier = nn.Identity()
        
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        return backbone, feature_dim

class LatLongDataset(Dataset):
    """Dataset for loading images and their lat-long coordinates."""

    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Get unique region IDs for embedding
        self.region_ids = sorted(self.data_frame['Region_ID'].unique())
        self.region_to_idx = {region: idx for idx,
                              region in enumerate(self.region_ids)}
        self.num_regions = len(self.region_ids)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image path
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0])

        # Load image
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            # Return a placeholder image in case of error
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get latitude and longitude (already scaled to [0,1])
        latitude = self.data_frame.iloc[idx, 1].astype(np.float32)
        longitude = self.data_frame.iloc[idx, 2].astype(np.float32)

        # Get angle and convert to sine and cosine components
        angle_deg = self.data_frame.iloc[idx, 3].astype(np.float32)
        angle_rad = angle_deg * math.pi / 180.0
        angle_sin = math.sin(angle_rad)
        angle_cos = math.cos(angle_rad)

        # Get Region_ID and convert to integer index
        region_id = self.data_frame.iloc[idx, 4]
        region_idx = self.region_to_idx[region_id]

        # convert to torch tensors
        angle_sin = torch.tensor(angle_sin, dtype=torch.float32)
        angle_cos = torch.tensor(angle_cos, dtype=torch.float32)

        # region index as long
        region_idx = torch.tensor(region_idx, dtype=torch.long)

        # latitude & longitude too
        latitude = torch.tensor(latitude,  dtype=torch.float32)
        longitude = torch.tensor(longitude, dtype=torch.float32)

        # Create sample
        sample = {
            'image': image,
            'latitude': latitude,
            'longitude': longitude,
            'angle_sin': angle_sin,
            'angle_cos': angle_cos,
            'region_idx': region_idx,
            # Include filename for evaluation
            'filename': self.data_frame.iloc[idx, 0]
        }

        return sample

class GeoRegressionModel(nn.Module):
    """Model for regressing latitude and longitude from images."""

    def __init__(self, backbone_name, num_regions, embedding_dim=16, dropout_rate=0.3, 
                 pretrained=True, head_hidden_dim=128):
        """
        Args:
            backbone_name (string): Name of the backbone CNN model.
            num_regions (int): Number of unique regions for embedding.
            embedding_dim (int): Dimension of the region embedding.
            dropout_rate (float): Dropout rate for the regression heads.
            pretrained (bool): Whether to use pretrained weights.
            head_hidden_dim (int): Hidden dimension size for regression heads.
        """
        super(GeoRegressionModel, self).__init__()

        # Initialize backbone CNN using the factory
        self.backbone_name = backbone_name
        self.backbone, self.feature_dim = BackboneFactory.get_backbone(backbone_name, pretrained)

        # Region embedding
        self.region_embedding = nn.Embedding(num_regions, embedding_dim)

        # Combined feature dimension (CNN features + region embedding + angle sine & cosine)
        combined_dim = self.feature_dim + embedding_dim + 2  # +2 for sin and cos of angle

        # Regression heads for latitude and longitude with configurable hidden dimension
        self.head_lat = nn.Sequential(
            nn.Linear(combined_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(head_hidden_dim, head_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(head_hidden_dim // 2, 1)
        )

        self.head_lon = nn.Sequential(
            nn.Linear(combined_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(head_hidden_dim, head_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(head_hidden_dim // 2, 1)
        )

    def forward(self, image, region_idx, angle_sin, angle_cos):
        # Extract image features
        x = self.backbone(image)
        
        # Handle different output formats from different backbones
        if len(x.shape) == 4:  # If output is [B, C, H, W]
            x = torch.flatten(x, 1)  # Flatten to [B, C*H*W]
        
        # Get region embeddings
        region_embed = self.region_embedding(region_idx)

        # Prepare angle features
        angle_features = torch.stack([angle_sin, angle_cos], dim=1)

        # Concatenate all features
        combined_features = torch.cat([x, region_embed, angle_features], dim=1)

        # Predict latitude and longitude
        latitude = self.head_lat(combined_features).squeeze(1)
        longitude = self.head_lon(combined_features).squeeze(1)

        return latitude, longitude

def get_transforms(is_train=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    if is_train:
        return transforms.Compose([
            # --- PIL-based augmentations ---
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            # --- now convert to tensor ---
            transforms.ToTensor(),
            normalize,
            # --- tensor-based transforms only ---
            transforms.RandomAffine(degrees=15,
                                    translate=(0.1, 0.1),
                                    scale=(0.9, 1.1),
                                    shear=10),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0

    for batch in tqdm(train_loader, desc="Training"):
        # Get data
        images = batch['image'].to(device)
        latitudes = batch['latitude'].to(device)
        longitudes = batch['longitude'].to(device)
        angle_sins = batch['angle_sin'].to(device)
        angle_coss = batch['angle_cos'].to(device)
        region_idxs = batch['region_idx'].to(device)

        # Forward pass
        optimizer.zero_grad()
        pred_lats, pred_lons = model(
            images, region_idxs, angle_sins, angle_coss)

        # TODO: can also propogate back by fixing head one at a time

        # Compute loss
        loss_lat = criterion(pred_lats, latitudes)
        loss_lon = criterion(pred_lons, longitudes)
        loss = (loss_lat + loss_lon) / 2.0  # Average of the two MSEs

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def validate(model, val_loader, criterion, device, lat_scalar, long_scalar):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    running_loss_scaled = 0.0
    all_pred_lats = []
    all_pred_lons = []
    all_true_lats = []
    all_true_lons = []

    scale_lat_tensor = lambda x : torch.from_numpy(lat_scalar.inverse_transform(x.cpu().numpy().reshape(-1, 1))).to(device=device)
    scale_lon_tensor = lambda x: torch.from_numpy(long_scalar.inverse_transform(x.cpu().numpy().reshape(-1, 1))).to(device=device)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            # Get data
            images = batch['image'].to(device)
            latitudes = batch['latitude'].to(device)
            longitudes = batch['longitude'].to(device)
            angle_sins = batch['angle_sin'].to(device)
            angle_coss = batch['angle_cos'].to(device)
            region_idxs = batch['region_idx'].to(device)

            # Forward pass
            pred_lats, pred_lons = model(
                images, region_idxs, angle_sins, angle_coss)


            # Get Scaled predictions
            scaled_pred_lat = scale_lat_tensor(pred_lats)
            scaled_pred_lons = scale_lon_tensor(pred_lons)

            scaled_lat = scale_lat_tensor(latitudes)
            scaled_lons = scale_lon_tensor(longitudes)

            # Compute loss
            loss_lat = criterion(pred_lats, latitudes)
            loss_lon = criterion(pred_lons, longitudes)
            loss = (loss_lat + loss_lon) / 2.0

            loss_scaled_lat = criterion(scaled_pred_lat, scaled_lat)
            loss_scaled_lon = criterion(scaled_pred_lons, scaled_lons)
            loss_scaled = (loss_scaled_lat + loss_scaled_lon) / 2.0

            running_loss += loss.item() * images.size(0)
            running_loss_scaled += loss_scaled.item() * images.size(0)

            # Store predictions and true values for metrics
            all_pred_lats.extend(pred_lats.cpu().numpy())
            all_pred_lons.extend(pred_lons.cpu().numpy())
            all_true_lats.extend(latitudes.cpu().numpy())
            all_true_lons.extend(longitudes.cpu().numpy())

    # Calculate overall validation loss
    val_loss = running_loss / len(val_loader.dataset)
    val_loss_scaled = running_loss_scaled / len(val_loader.dataset)

    # Calculate MSE for latitude and longitude separately
    lat_mse = mean_squared_error(all_true_lats, all_pred_lats)
    lon_mse = mean_squared_error(all_true_lons, all_pred_lons)

    # Convert collected lists to NumPy arrays and inverse‐scale once
    true_lats_np = np.array(all_true_lats).reshape(-1, 1)
    pred_lats_np = np.array(all_pred_lats).reshape(-1, 1)
    true_lons_np = np.array(all_true_lons).reshape(-1, 1)
    pred_lons_np = np.array(all_pred_lons).reshape(-1, 1)

    scaled_true_lats = lat_scalar.inverse_transform(true_lats_np).flatten()
    scaled_pred_lats = lat_scalar.inverse_transform(pred_lats_np).flatten()
    scaled_true_lons = long_scalar.inverse_transform(true_lons_np).flatten()
    scaled_pred_lons = long_scalar.inverse_transform(pred_lons_np).flatten()

    lat_scaled_mse = mean_squared_error(scaled_true_lats, scaled_pred_lats)
    lon_scaled_mse = mean_squared_error(scaled_true_lons, scaled_pred_lons)

    # Calculate combined MSE
    combined_mse = (lat_mse + lon_mse) / 2.0

    return val_loss, lat_mse, lon_mse, combined_mse, val_loss_scaled, lat_scaled_mse, lon_scaled_mse


def save_checkpoint(model, optimizer, scheduler, epoch, val_mse, filename):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_mse': val_mse
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def plot_metrics(train_losses, val_losses, lat_mses, lon_mses, combined_mses, scaled_lat_mses, scaled_lon_mses, save_path):
    """Plot training and validation metrics."""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(20, 10))

    # Plot losses
    plt.subplot(3, 1, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot MSEs
    plt.subplot(3, 1, 2)
    plt.plot(epochs, lat_mses, 'g-', label='Latitude MSE')
    plt.plot(epochs, lon_mses, 'y-', label='Longitude MSE')
    plt.plot(epochs, combined_mses, 'c-', label='Combined MSE')
    plt.title('MSE Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()

    # Plot scaled MSEs
    plt.subplot(3, 1, 3)
    plt.plot(epochs, scaled_lat_mses, 'g-', label='Latitude MSE')
    plt.plot(epochs, scaled_lon_mses, 'y-', label='Longitude MSE')
    plt.title('Scaled MSE Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Scaled MSE')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load in Latitude and Longitude Scalar
    lat_scalar = joblib.load(args.lat_scalar_path)
    lon_scalar = joblib.load(args.lon_scalar_path)

    # Setup data paths
    train_csv = os.path.join(args.data_dir, "train_label.csv")
    val_csv = os.path.join(args.data_dir, "val_label.csv")
    train_img_dir = os.path.join(args.data_dir, "images_train")
    val_img_dir = os.path.join(args.data_dir, "images_val")

    # Create datasets
    train_dataset = LatLongDataset(
        csv_file=train_csv,
        img_dir=train_img_dir,
        transform=get_transforms(is_train=True)
    )

    val_dataset = LatLongDataset(
        csv_file=val_csv,
        img_dir=val_img_dir,
        transform=get_transforms(is_train=False)
    )

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of unique regions: {train_dataset.num_regions}")

    # Create data loaders
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

    # Create model
    model = GeoRegressionModel(
        backbone_name=args.backbone,
        num_regions=train_dataset.num_regions,
        embedding_dim=args.embedding_dim,
        dropout_rate=args.dropout_rate,
        pretrained=True
    )
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=args.lr_patience,
    )

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize variables for early stopping and metrics tracking
    best_val_mse = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    lat_mses = []
    lon_mses = []
    combined_mses = []
    scaled_lat_mses = []
    scaled_lon_mses = []

    model_name = f"{args.backbone}_emb{args.embedding_dim}_dr{args.dropout_rate}"

    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train one epoch
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, lat_mse, lon_mse, combined_mse, scaled_val_loss, scaled_lat_loss, scaled_lon_loss = validate(
            model, val_loader, criterion, device, lat_scalar, lon_scalar)

        # Update scheduler
        scheduler.step(val_loss)

        # Print metrics
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}, Scaled Val Loss: {scaled_val_loss:.6f}")
        print(
            f"Lat MSE: {lat_mse:.6f}, Lon MSE: {lon_mse:.6f}, Combined MSE: {combined_mse:.6f}")
        print(
            f"Scaled Lat MSE: {scaled_lat_loss:.2f},Scaled Lon MSE: {scaled_lon_loss:.2f}")

        # Track metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        lat_mses.append(lat_mse)
        lon_mses.append(lon_mse)
        combined_mses.append(combined_mse)
        scaled_lat_mses.append(scaled_lat_loss)
        scaled_lon_mses.append(scaled_lon_loss)

        # # Save checkpoint
        # checkpoint_path = os.path.join(args.output_dir, f"{model_name}_epoch{epoch}.pth")
        # save_checkpoint(model, optimizer, scheduler, epoch, combined_mse, checkpoint_path)

        # Check for early stopping
        if combined_mse < best_val_mse:
            best_val_mse = combined_mse
            patience_counter = 0

            # Save best model
            best_model_path = os.path.join(
                args.output_dir, f"{model_name}.pth")
            save_checkpoint(model, optimizer, scheduler, epoch,
                            best_val_mse, best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # Plot metrics
    plot_path = os.path.join(args.output_dir, f"{model_name}_metrics.png")
    plot_metrics(train_losses, val_losses, lat_mses,
                 lon_mses, combined_mses, scaled_lat_mses, scaled_lon_mses, plot_path)

    # Final evaluation
    print("\nFinal Evaluation on Validation Set:")
    _, final_lat_mse, final_lon_mse, final_combined_mse, scaled_val_loss, scaled_lat_loss, scaled_lon_loss= validate(
        model, val_loader, criterion, device, lat_scalar, lon_scalar)
    print(f"Final Lat MSE: {final_lat_mse:.6f}(Unscaled) ,{scaled_lat_loss:.2f}(Scaled)")
    print(f"Final Lon MSE: {final_lon_mse:.6f}(Unscaled) ,{scaled_lon_loss:.2f}(Scaled)")
    print(f"Final Combined MSE: {final_combined_mse:.6f}")
    print(f"Final Scaled Val MSE: {scaled_val_loss:.2f}")

    # Save results to a summary file
    summary_path = os.path.join(args.output_dir, f"{model_name}_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Model: {args.backbone}\n")
        f.write(f"Embedding Dimension: {args.embedding_dim}\n")
        f.write(f"Dropout Rate: {args.dropout_rate}\n")
        f.write(f"Learning Rate: {args.learning_rate}\n")
        f.write(f"Weight Decay: {args.weight_decay}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Best Validation MSE: {best_val_mse:.6f}\n")
        f.write(f"Final Latitude MSE: {final_lat_mse:.6f}\n")
        f.write(f"Final Longitude MSE: {final_lon_mse:.6f}\n")
        f.write(f"Final Combined MSE: {final_combined_mse:.6f}\n")

    print(f"Training completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model for latitude and longitude regression.")

    # Data parameters
    parser.add_argument("--data_dir", type=str, default="./LatLongDataset",
                        help="Path to the dataset directory")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save model checkpoints and results")
    parser.add_argument("--lat_scalar_path", type=str, default="lat_scaler.pkl", help="Path to latitude scalar object file")
    parser.add_argument("--lon_scalar_path", type=str, default="long_scaler.pkl", help="Path to longitude scalar object file")

    # Model parameters
    parser.add_argument("--backbone", type=str, default="resnet34",
                        choices=["resnet18", "resnet34", "resnet50", "resnet101",
                                 "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
                                 "mobilenet_v3_small", "mobilenet_v3_large",
                                 "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32",
                                 "swin_t", "swin_s", "swin_b",
                                 "convnext_tiny", "convnext_small", "convnext_base",
                                 "regnet_y_400mf", "regnet_y_800mf", "regnet_y_1_6gf",
                                 "densenet121", "densenet169",
                                 "maxvit_t"],
                        help="CNN backbone architecture")
    parser.add_argument("--embedding_dim", type=int, default=16,
                        help="Dimension of the region embedding")
    parser.add_argument("--dropout_rate", type=float, default=0.3,
                        help="Dropout rate for the regression heads")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and validation")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay (L2 regularization)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Maximum number of epochs to train")
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                        help="Number of epochs to wait before early stopping")
    parser.add_argument("--lr_patience", type=int, default=5,
                        help="Number of epochs to wait before reducing learning rate")
    parser.add_argument("--num_workers", type=int, default=10,
                        help="Number of worker processes for data loading")

    args = parser.parse_args()

    main(args)
