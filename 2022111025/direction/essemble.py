"""
Ensemble Orientation Predictor
------------------------------
This script builds an ensemble of trained image orientation models.
It calculates ensemble weights based on validation MAE, uses softmax with
tunable temperature, and makes predictions on test images.
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import pi

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from PIL import Image

# Import model definition from the training script
try:
    from angle import AnglePredictor
except ImportError:
    print("Error: Could not import from angle.py.")
    print("Please ensure that angle.py is in the current directory.")
    sys.exit(1)

class EnsemblePredictor:
    def __init__(self, models_dir, model_maes=None, temperature=1.0, embed_dim=16, num_region=15):
        """
        Initialize the ensemble predictor.
        
        Args:
            models_dir (str): Directory containing model subdirectories
            model_maes (dict, optional): Dictionary mapping model names to their MAE values
            temperature (float): Temperature parameter for softmax weighting
        """
        self.models_dir = models_dir
        self.temperature = temperature
        self.embed_dim = embed_dim
        self.num_regions = num_region
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Discover available models
        self.available_models = self._discover_models()
        print(f"Discovered {len(self.available_models)} models: {', '.join(self.available_models)}")
        
        # Set or request MAE values
        if model_maes is None:
            self.model_maes = self._request_model_maes()
        else:
            self.model_maes = model_maes
            
        # Calculate model weights using softmax
        self.model_weights = self._calculate_weights()
        
        # Load the models
        self.models = self._load_models()
    
    def _discover_models(self):
        """Discover available models in the models directory"""
        models = []
        for model_dir in os.listdir(self.models_dir):
            dir_path = os.path.join(self.models_dir, model_dir)
            if os.path.isdir(dir_path):
                # Check if there's a .pth file in this directory
                model_files = glob.glob(os.path.join(dir_path, "*.pth"))
                if model_files:
                    models.append(model_dir)
        return models
    
    def _request_model_maes(self):
        """Request MAE values for each model from user if not provided"""
        model_maes = {}
        print("Please enter the validation MAE for each model:")
        for model_name in self.available_models:
            while True:
                try:
                    mae = float(input(f"MAE for {model_name}: "))
                    if mae > 0:
                        model_maes[model_name] = mae
                        break
                    else:
                        print("MAE must be positive. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a numeric value.")
        return model_maes
    
    def _calculate_weights(self):
        """Calculate model weights based on MAE values using softmax with temperature"""
        # Lower MAE should get higher weight, so we use negative MAE
        if not self.model_maes:
            print("Error: No MAE values provided. Cannot calculate weights.")
            return {model: 1.0/len(self.available_models) for model in self.available_models}
        
        # Get MAE for available models
        maes = np.array([self.model_maes.get(model, 180.0) for model in self.available_models])  # Default to worst MAE (180°)
        
        # Convert MAE to scores (lower MAE = higher score)
        # We use 1/MAE as the score
        scores = 1.0 / (maes + 1e-6)  # Add small epsilon to avoid division by zero
        
        # Apply softmax with temperature
        weights = np.exp(scores / self.temperature)
        weights = weights / np.sum(weights)
        
        # Create a dictionary mapping model names to weights
        model_weights = {model: float(weight) for model, weight in zip(self.available_models, weights)}
        
        # Print the weights
        print("\nModel Weights:")
        for model, weight in model_weights.items():
            print(f"{model}: {weight:.4f}")
        
        return model_weights
    
    def _load_models(self):
        """Load all available models"""
        models = {}
        for model_name in self.available_models:
            try:
                # Extract backbone name from model directory
                backbone = model_name.split('_')[0] if '_' in model_name else model_name
                
                # Find model file
                model_path = os.path.join(self.models_dir, model_name)
                model_files = glob.glob(os.path.join(model_path, "*.pth"))
                
                if not model_files:
                    print(f"Warning: No .pth file found for {model_name}. Skipping.")
                    continue
                
                model_file = model_files[0]  # Use the first one if multiple files exist
                
                # Create model instance
                # We need to determine the number of regions dynamically
                # For simplicity, let's use a large number for the region embedding
                # The actual regions used will be limited by the training data
                model = AnglePredictor(backbone_name=model_name, num_regions=self.num_regions, embed_dim=self.embed_dim)
                
                # Load weights
                model.load_state_dict(torch.load(model_file, map_location=self.device))
                model.to(self.device)
                model.eval()
                
                models[model_name] = model
                print(f"Loaded model: {model_name}")
                
            except Exception as e:
                print(f"Error loading model {model_name}: {str(e)}")
                continue
                
        return models
    
    def predict(self, image_tensor, region_id):
        """
        Make a prediction using the ensemble.
        
        Args:
            image_tensor (torch.Tensor): Image tensor of shape [1, 3, 224, 224]
            region_id (torch.Tensor): Region ID tensor of shape [1]
            
        Returns:
            float: Predicted angle in degrees [0, 360)
        """
        if not self.models:
            raise ValueError("No models loaded. Cannot make predictions.")
            
        image_tensor = image_tensor.to(self.device)
        region_id = region_id.to(self.device)
        
        # Collect predictions from all models
        model_predictions = {}
        
        with torch.no_grad():
            for model_name, model in self.models.items():
                output = model(image_tensor, region_id)
                
                # Convert [cos, sin] to angle
                cos_val, sin_val = output[0].cpu().numpy()
                angle = np.degrees(np.arctan2(sin_val, cos_val)) % 360
                
                model_predictions[model_name] = angle
        
        # Calculate weighted average of angles (handling the circular nature)
        sin_sum = 0.0
        cos_sum = 0.0
        
        for model_name, angle in model_predictions.items():
            weight = self.model_weights.get(model_name, 0.0)
            angle_rad = np.radians(angle)
            sin_sum += weight * np.sin(angle_rad)
            cos_sum += weight * np.cos(angle_rad)
        
        # Calculate the average angle
        ensemble_angle = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360
        
        return ensemble_angle, model_predictions

class TestDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Dataset for test images.
        
        Args:
            csv_file (str): Path to the CSV file with image filenames and region IDs
            img_dir (str): Directory with the images
            transform (callable, optional): Optional transform to be applied on images
        """
        self.data = pd.read_csv(csv_file, dtype={"filename": str})
        self.img_dir = img_dir
        self.transform = transform
        
        # Extract unique region IDs and create a mapping
        if 'Region_ID' in self.data.columns:
            self.unique_regions = self.data['Region_ID'].unique()
            self.region_to_idx = {region: idx for idx, region in enumerate(self.unique_regions)}
        else:
            print("Warning: 'Region_ID' column not found in CSV. Using dummy region IDs.")
            self.unique_regions = [0]
            self.region_to_idx = {0: 0}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        fname = str(self.data.loc[idx, "filename"])
        img_path = os.path.join(self.img_dir, fname)
        image = Image.open(img_path).convert('RGB')
        
        # Get region ID if available, else use default
        region_id = 0
        if 'Region_ID' in self.data.columns:
            region_id = self.data.iloc[idx, 1]
            region_id = self.region_to_idx.get(region_id, 0)
        
        # Get true angle if available (for evaluation)
        true_angle = None
        if 'angle' in self.data.columns:
            true_angle = self.data.loc[idx, "angle"]
        else:
            true_angle = -1
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        filename = self.data.loc[idx, "filename"]
        
        return image, torch.tensor(region_id, dtype=torch.long), filename, true_angle

def predict_test_set(ensemble, test_loader, output_csv):
    """
    Make predictions on the test set using the ensemble.
    
    Args:
        ensemble (EnsemblePredictor): The ensemble predictor
        test_loader (DataLoader): DataLoader for test images
        output_csv (str): Path to output CSV file
    """
    results = []
    has_true_angles = False
    
    for images, region_ids, filenames, true_angles in tqdm(test_loader, desc="Predicting"):
        for i in range(len(images)):
            image = images[i].unsqueeze(0)  # Add batch dimension
            region_id = region_ids[i].unsqueeze(0)
            filename = filenames[i]
            
            # Get ensemble prediction
            pred_angle, model_predictions = ensemble.predict(image, region_id)
            
            # Convert predictions from tensor to scalar
            pred_angle = pred_angle.item()
            
            # Store result
            result = {'filename': filename, 'predicted_angle': pred_angle}
            
            # # Add individual model predictions
            # for model_name, angle in model_predictions.items():
            #     result[f'{model_name}_pred'] = angle
            
            # Add true angle if available (for evaluation)
            if true_angles[i] != -1:
                has_true_angles = True
                result['true_angle'] = true_angles[i].item()
                # Calculate error
                error = min(abs(pred_angle - true_angles[i]), 360 - abs(pred_angle - true_angles[i]))
                result['error'] = error.item()
            
            results.append(result)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")
    
    # If true angles are available, calculate MAE
    if has_true_angles:
        mae = df['error'].mean()
        print(f"Ensemble Mean Absolute Angular Error: {mae:.2f}°")
        
        # Create visualization of errors
        plt.figure(figsize=(10, 6))
        plt.hist(df['error'], bins=36, alpha=0.7)
        plt.axvline(mae, color='r', linestyle='--', label=f'MAE: {mae:.2f}°')
        plt.xlabel('Angular Error (degrees)')
        plt.ylabel('Count')
        plt.title('Ensemble Prediction Error Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('ensemble_error_distribution.png')
        plt.close()
        
        # Save a few examples with largest errors
        top_errors = df.sort_values('error', ascending=False).head(10)
        print("\nTop 10 largest errors:")
        print(top_errors[['filename', 'true_angle', 'predicted_angle', 'error']])

def main(args):
    # Define data transformations for test images
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Create test dataset and dataloader
    test_dataset = TestDataset(
        csv_file=args.test_csv,
        img_dir=args.test_dir,
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Parse MAE values if provided as command-line arguments
    model_maes = None
    if args.model_maes:
        model_maes = {}
        mae_pairs = args.model_maes.split(',')
        for pair in mae_pairs:
            if ':' in pair:
                model_name, mae_str = pair.split(':')
                try:
                    mae = float(mae_str)
                    model_maes[model_name] = mae
                except ValueError:
                    print(f"Warning: Invalid MAE value for {model_name}: {mae_str}")
    
    # Create ensemble predictor
    ensemble = EnsemblePredictor(
        models_dir=args.models_dir,
        model_maes=model_maes,
        temperature=args.temperature,
        embed_dim=args.embed_dim,
        num_region=args.num_region,
    )
    
    # Make predictions on test set
    predict_test_set(ensemble, test_loader, args.output_csv)
    
    print("\nEnsemble prediction completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble Orientation Predictor")
    
    # Required arguments
    parser.add_argument("--models_dir", type=str, default="outputs",
                        help="Directory containing model subdirectories")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Directory containing test images")
    parser.add_argument("--test_csv", type=str, required=True,
                        help="CSV file with test image filenames and region IDs")
    parser.add_argument("--output_csv", type=str, default="Angle_test.csv",
                        help="Output CSV file for predictions")
    
    # Optional arguments
    parser.add_argument("--model_maes", type=str, default=None,
                        help="Comma-separated list of model:MAE pairs (e.g., 'resnet18:10.5,efficientnet_b0:8.3')")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature parameter for softmax weighting (lower values make weights more extreme)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for test predictions")
    parser.add_argument("--num_region", type=int, default=15,
                        help="Number of regions for prediction")
    parser.add_argument("--embed_dim", type=int, default=16,
                        help="Dimensionality of the region embedding")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() - 2,
                        help="Number of worker threads for data loading")
    
    args = parser.parse_args()
    
    main(args)