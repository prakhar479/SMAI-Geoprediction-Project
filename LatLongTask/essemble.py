#!/usr/bin/env python3
import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image, ImageFile
import joblib
from tqdm import tqdm
import json
from collections import defaultdict

# Import necessary model definitions
from torchvision import models

# Ensure PIL can load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class LatLongDataset(torch.utils.data.Dataset):
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
        self.region_to_idx = {region: idx for idx, region in enumerate(self.region_ids)}
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

        # Get angle and convert to sine and cosine components
        angle_deg = self.data_frame.iloc[idx, 3].astype(np.float32)
        angle_rad = angle_deg * np.pi / 180.0
        angle_sin = np.sin(angle_rad)
        angle_cos = np.cos(angle_rad)

        # Get Region_ID and convert to integer index
        region_id = self.data_frame.iloc[idx, 4]
        region_idx = self.region_to_idx[region_id]

        # Convert to torch tensors
        angle_sin = torch.tensor(angle_sin, dtype=torch.float32)
        angle_cos = torch.tensor(angle_cos, dtype=torch.float32)
        region_idx = torch.tensor(region_idx, dtype=torch.long)

        # Create sample
        sample = {
            'image': image,
            'angle_sin': angle_sin,
            'angle_cos': angle_cos,
            'region_idx': region_idx,
            'filename': self.data_frame.iloc[idx, 0]  # Include filename for evaluation
        }

        # Include ground truth if available (for evaluation)
        if len(self.data_frame.columns) > 5:
            latitude = self.data_frame.iloc[idx, 1].astype(np.float32)
            longitude = self.data_frame.iloc[idx, 2].astype(np.float32)
            sample['latitude'] = torch.tensor(latitude, dtype=torch.float32)
            sample['longitude'] = torch.tensor(longitude, dtype=torch.float32)

        return sample


class GeoRegressionModel(nn.Module):
    """Model for regressing latitude and longitude from images."""

    def __init__(self, backbone_name, num_regions, embedding_dim=16, dropout_rate=0.3):
        """
        Args:
            backbone_name (string): Name of the backbone CNN model.
            num_regions (int): Number of unique regions for embedding.
            embedding_dim (int): Dimension of the region embedding.
            dropout_rate (float): Dropout rate for the regression heads.
        """
        super(GeoRegressionModel, self).__init__()

        # Initialize backbone CNN
        self.backbone_name = backbone_name
        if backbone_name == 'resnet34':
            self.backbone = models.resnet34(weights=None)
            self.feature_dim = self.backbone.fc.in_features
            # Remove the final FC layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif backbone_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights=None)
            self.feature_dim = self.backbone.classifier[1].in_features
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # Remove the classifier
        elif backbone_name == 'mobilenet_v3_small':
            self.backbone = models.mobilenet_v3_small(weights=None)
            self.feature_dim = self.backbone.classifier[0].in_features
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # Remove the classifier
        elif backbone_name == 'vit_b_16':
            self.backbone = models.vit_b_16(weights=None)
            self.feature_dim = self.backbone.heads.head.in_features
            self.backbone.heads = nn.Identity()
        elif backbone_name == 'swin_b':
            self.backbone = models.swin_b(weights=None)
            self.feature_dim = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Region embedding
        self.region_embedding = nn.Embedding(num_regions, embedding_dim)

        # Combined feature dimension (CNN features + region embedding + angle sine & cosine)
        combined_dim = self.feature_dim + embedding_dim + 2  # +2 for sin and cos of angle

        # Regression heads for latitude and longitude
        self.head_lat = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )

        self.head_lon = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )

    def forward(self, image, region_idx, angle_sin, angle_cos):
        # Extract image features
        x = self.backbone(image)
        x = x.view(x.size(0), -1)  # Flatten features

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


class EnsembleModel:
    def __init__(self, models_dir, temperature=1.0, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.models = []
        self.model_infos = []
        self.device = device
        self.temperature = temperature
        
        # Load models
        self._load_models(models_dir)
        
        if not self.models:
            raise ValueError("No models were found in the specified directory")
    
        self.weights = self._compute_weights()
        
        print(f"Loaded {len(self.models)} models for ensemble")

    def _load_models(self, models_dir):
        # Find all model directories
        model_types = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        
        for model_type in model_types:
            model_type_dir = os.path.join(models_dir, model_type)
            model_files = glob.glob(os.path.join(model_type_dir, "*.pth"))
            
            for model_file in model_files:
                try:
                    # Parse model info from filename
                    filename = os.path.basename(model_file)
                    match = re.match(r'(.+)_emb(\d+)_dr([\d\.]+)\.pth', filename)
                    
                    if not match:
                        print(f"Skipping {filename} - doesn't match expected format")
                        continue
                    
                    backbone_name = match.group(1)
                    embedding_dim = int(match.group(2))
                    dropout_rate = float(match.group(3))
                    
                    # Load summary file to get validation MSE for weighting
                    summary_file = model_file.replace('.pth', '_summary.txt')
                    val_mse = float('inf')
                    
                    if os.path.exists(summary_file):
                        with open(summary_file, 'r') as f:
                            for line in f:
                                if 'Best Validation MSE' in line:
                                    val_mse = float(line.split(':')[1].strip())
                                    break
                    
                    # Load checkpoint
                    checkpoint = torch.load(model_file, map_location=self.device)
                    
                    # Create model
                    model = GeoRegressionModel(
                        backbone_name=backbone_name,
                        num_regions=15, 
                        embedding_dim=embedding_dim,
                        dropout_rate=dropout_rate
                    )
                    
                    # Try to load state dict with compatibility
                    try:
                        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    except Exception as e:
                        print(f"Error loading model {model_file}: {e}")
                        continue
                    
                    model = model.to(self.device)
                    model.eval()
                    
                    # Store model and its info
                    self.models.append(model)
                    self.model_infos.append({
                        'backbone': backbone_name,
                        'embedding_dim': embedding_dim,
                        'dropout_rate': dropout_rate,
                        'val_mse': val_mse,
                        'weight': 1.0  # Will be updated based on validation MSE
                    })
                    
                    # print(f"Loaded model: {backbone_name}, emb={embedding_dim}, dr={dropout_rate}, val_mse={val_mse}")
                
                except Exception as e:
                    print(f"Error processing model {model_file}: {e}")
    
    def _compute_weights(self):
        """Compute weights using temperature-scaled softmax based on validation MSE"""
        mse_values = np.array([info['val_mse'] for info in self.model_infos])
        
        # Convert MSE to accuracy-like metric (higher is better)
        # Using negative MSE since lower MSE is better
        scores = -mse_values
        
        # Apply temperature scaling and softmax
        scores = scores / self.temperature
        exp_scores = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
        weights = exp_scores / np.sum(exp_scores)
        
        # Update model weights
        for i, info in enumerate(self.model_infos):
            info['weight'] = float(weights[i])
            print(f"Model {info['backbone']} (emb={info['embedding_dim']}): weight = {weights[i]:.4f}")
        
        return weights
    
    def predict(self, images, region_idxs, angle_sins, angle_coss):
        """Make ensemble prediction"""
        # Compute weights if not already calculated
        weights = self.weights
        
        all_lat_preds = []
        all_lon_preds = []
        
        # Get predictions from each model
        for i, model in enumerate(self.models):
            with torch.no_grad():
                lat_preds, lon_preds = model(images, region_idxs, angle_sins, angle_coss)
                
                all_lat_preds.append(lat_preds.unsqueeze(0))
                all_lon_preds.append(lon_preds.unsqueeze(0))
        
        # Stack predictions
        all_lat_preds = torch.cat(all_lat_preds, dim=0)  # Shape: [num_models, batch_size]
        all_lon_preds = torch.cat(all_lon_preds, dim=0)  # Shape: [num_models, batch_size]
        
        # Apply weights
        weights_tensor = torch.tensor(weights, device=self.device).view(-1, 1)
        weighted_lat_preds = (all_lat_preds * weights_tensor).sum(dim=0)
        weighted_lon_preds = (all_lon_preds * weights_tensor).sum(dim=0)
        
        return weighted_lat_preds, weighted_lon_preds


def get_transforms():
    """Get transforms for prediction"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])


def predict(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading test data from {args.test_csv}...")
    test_dataset = LatLongDataset(
        csv_file=args.test_csv,
        img_dir=args.test_img_dir,
        transform=get_transforms()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Number of test samples: {len(test_dataset)}")
    print(f"Number of unique regions: {test_dataset.num_regions}")
    
    # Load ensemble model
    print(f"Creating ensemble model with temperature {args.temperature}...")
    ensemble = EnsembleModel(args.models_dir, temperature=args.temperature, device=device)
    
    # Load scalers
    print(f"Loading scalers...")
    lat_scalar = joblib.load(args.lat_scalar_path)
    long_scalar = joblib.load(args.long_scalar_path)
    
    # Initialize variables for predictions
    all_filenames = []
    all_lat_preds = []
    all_lon_preds = []
    all_true_lats = []
    all_true_lons = []
    has_ground_truth = False
    
    # Make predictions
    print("Making predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            # Get data
            images = batch['image'].to(device)
            angle_sins = batch['angle_sin'].to(device)
            angle_coss = batch['angle_cos'].to(device)
            region_idxs = batch['region_idx'].to(device)
            filenames = batch['filename']
            
            # Check if ground truth is available
            if 'latitude' in batch and 'longitude' in batch:
                has_ground_truth = True
                true_lats = batch['latitude'].cpu().numpy()
                true_lons = batch['longitude'].cpu().numpy()
                all_true_lats.extend(true_lats)
                all_true_lons.extend(true_lons)
            
            # Forward pass with ensemble
            pred_lats, pred_lons = ensemble.predict(images, region_idxs, angle_sins, angle_coss)
            
            # Store predictions
            all_filenames.extend(filenames)
            all_lat_preds.extend(pred_lats.cpu().numpy())
            all_lon_preds.extend(pred_lons.cpu().numpy())
    
    # Create DataFrame with predictions
    results_df = pd.DataFrame({
        'filename': all_filenames,
        'pred_latitude_scaled': all_lat_preds,
        'pred_longitude_scaled': all_lon_preds
    })
    
    # Inverse transform the scaled predictions
    pred_lats_np = np.array(all_lat_preds).reshape(-1, 1)
    pred_lons_np = np.array(all_lon_preds).reshape(-1, 1)
    
    unscaled_pred_lats = lat_scalar.inverse_transform(pred_lats_np).flatten()
    unscaled_pred_lons = long_scalar.inverse_transform(pred_lons_np).flatten()
    
    results_df['pred_latitude'] = unscaled_pred_lats
    results_df['pred_longitude'] = unscaled_pred_lons
    
    # Add ground truth if available
    if has_ground_truth:
        results_df['true_latitude_scaled'] = all_true_lats
        results_df['true_longitude_scaled'] = all_true_lons
        
        true_lats_np = np.array(all_true_lats).reshape(-1, 1)
        true_lons_np = np.array(all_true_lons).reshape(-1, 1)
        
        unscaled_true_lats = lat_scalar.inverse_transform(true_lats_np).flatten()
        unscaled_true_lons = long_scalar.inverse_transform(true_lons_np).flatten()
        
        results_df['true_latitude'] = unscaled_true_lats
        results_df['true_longitude'] = unscaled_true_lons
        
        # Calculate errors
        scaled_lat_mse = np.mean((np.array(all_lat_preds) - np.array(all_true_lats)) ** 2)
        scaled_lon_mse = np.mean((np.array(all_lon_preds) - np.array(all_true_lons)) ** 2)
        scaled_combined_mse = (scaled_lat_mse + scaled_lon_mse) / 2
        
        unscaled_lat_mse = np.mean((unscaled_pred_lats - unscaled_true_lats) ** 2)
        unscaled_lon_mse = np.mean((unscaled_pred_lons - unscaled_true_lons) ** 2)
        unscaled_combined_mse = (unscaled_lat_mse + unscaled_lon_mse) / 2
        
        print("\nEvaluation Results:")
        print(f"Scaled Latitude MSE: {scaled_lat_mse:.6f}")
        print(f"Scaled Longitude MSE: {scaled_lon_mse:.6f}")
        print(f"Scaled Combined MSE: {scaled_combined_mse:.6f}")
        print(f"Unscaled Latitude MSE: {unscaled_lat_mse:.6f}")
        print(f"Unscaled Longitude MSE: {unscaled_lon_mse:.6f}")
        print(f"Unscaled Combined MSE: {unscaled_combined_mse:.6f}")
        
        # Save evaluation results
        eval_results = {
            'scaled_lat_mse': float(scaled_lat_mse),
            'scaled_lon_mse': float(scaled_lon_mse),
            'scaled_combined_mse': float(scaled_combined_mse),
            'unscaled_lat_mse': float(unscaled_lat_mse),
            'unscaled_lon_mse': float(unscaled_lon_mse),
            'unscaled_combined_mse': float(unscaled_combined_mse),
            'ensemble_weights': {f"{info['backbone']}_emb{info['embedding_dim']}_dr{info['dropout_rate']}": info['weight'] 
                                for info in ensemble.model_infos},
            'temperature': args.temperature
        }
        
        with open(args.output_eval, 'w') as f:
            json.dump(eval_results, f, indent=4)
        
        print(f"Evaluation results saved to {args.output_eval}")
    
    # Save predictions
    results_df.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions with an ensemble of geo-regression models.")
    
    # Input/output parameters
    parser.add_argument("--test_csv", type=str, required=True,
                        help="Path to the test CSV file with filenames and metadata")
    parser.add_argument("--test_img_dir", type=str, required=True,
                        help="Directory containing test images")
    parser.add_argument("--models_dir", type=str, default="outputs",
                        help="Directory containing model checkpoints")
    parser.add_argument("--output_csv", type=str, default="LatLongPrediction.csv",
                        help="Path to save prediction results")
    parser.add_argument("--output_eval", type=str, default="ensemble_evaluation.json",
                        help="Path to save evaluation results (if ground truth available)")
    parser.add_argument("--lat_scalar_path", type=str, default="lat_scaler.pkl",
                        help="Path to latitude scalar object file")
    parser.add_argument("--long_scalar_path", type=str, default="long_scaler.pkl",
                        help="Path to longitude scalar object file")
    
    # Prediction parameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for prediction")
    parser.add_argument("--temperature", type=float, default=0.25,
                        help="Temperature for softmax weighting (lower = more confident models get higher weight)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker processes for data loading")
    
    args = parser.parse_args()
    
    predict(args)