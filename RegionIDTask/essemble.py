#!/usr/bin/env python3
"""
Ensemble Prediction Script for Region-Based Image Classification

This script loads multiple trained models, creates an ensemble based on their validation accuracies,
and generates predictions for a test set.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create argument parser
parser = argparse.ArgumentParser(description='Ensemble prediction for region classifier models')
parser.add_argument('--models_dir', type=str, default='outputs', help='Directory containing model folders')
parser.add_argument('--test_dir', type=str, required=True, help='Directory containing test images')
parser.add_argument('--output_csv', type=str, default='RegionId_test.csv', help='Output CSV filename')
parser.add_argument('--num_classes', type=int, default=15, help='Number of region classes')
parser.add_argument('--temp', type=float, default=1.0, help='Temperature for softmax scaling')
parser.add_argument('--chkpt', type=str, default='final_model.pth', help='Checkpoint filename to load')
args = parser.parse_args()

def get_transforms():
    """Create data transformations for validation/test data."""
    # ImageNet normalization values
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Validation/Test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    return val_transform

def create_model(model_type, num_classes):
    """Create a model with a new classification head."""
    # EffcientNet models
    if model_type == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
    elif model_type == 'efficientnet_b1':
        model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
    elif model_type == 'efficientnet_b2':
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
    elif model_type == 'efficientnet_b3':
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
    # MobileNet models
    elif model_type == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        num_ftrs = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
    elif model_type == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        num_ftrs = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
    # ResNet models
    elif model_type == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
    elif model_type == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
    elif model_type == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
    elif model_type == 'resnet101':
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
    # ConvNeXt models
    elif model_type == 'convnext_tiny':
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, num_classes)
    elif model_type == 'convnext_small':
        model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, num_classes)
    elif model_type == 'convnext_base':
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, num_classes)
    # Vision Transformer models
    elif model_type == 'vit_b_16':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        num_ftrs = model.heads.head.in_features
        model.heads = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
    elif model_type == 'vit_b_32':
        model = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT)
        num_ftrs = model.heads.head.in_features
        model.heads = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
    elif model_type == 'vit_l_16':
        model = models.vit_l_16(weights=models.ViT_L_16_Weights.DEFAULT)
        num_ftrs = model.heads.head.in_features
        model.heads = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
    # Swin Transformer models
    elif model_type == 'swin_t':
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        num_ftrs = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
    elif model_type == 'swin_s':
        model = models.swin_s(weights=models.Swin_S_Weights.DEFAULT)
        num_ftrs = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
    elif model_type == 'swin_b':
        model = models.swin_b(weights=models.Swin_B_Weights.DEFAULT)
        num_ftrs = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model.to(device)

def load_model_with_weights(model_type, model_path, num_classes):
    """Load a model and its weights from the given path."""
    model = create_model(model_type, num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # If it's a full checkpoint with model_state_dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        # Get validation accuracy if available
        val_acc = checkpoint.get('val_acc', 0.0)
    else:
        # If it's just the model weights
        model.load_state_dict(checkpoint)
        val_acc = 0.0  # Default if not available
        
    print(f"Loaded {model_type} from {model_path} with validation accuracy: {val_acc:.2f}%")
    return model, val_acc

def find_trained_models(models_dir):
    """Find all trained models in the directory and extract their validation accuracies."""
    models_info = []
    
    # List all model directories
    model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    
    for model_dir in model_dirs:
        model_type = model_dir  # Directory name is the model type
        model_path = os.path.join(models_dir, model_dir, args.chkpt)
        
        if os.path.exists(model_path):
            try:
                # Load checkpoint to extract validation accuracy
                checkpoint = torch.load(model_path, map_location=device)
                if 'val_acc' in checkpoint:
                    val_acc = checkpoint['val_acc']
                    models_info.append({
                        'model_type': model_type,
                        'model_path': model_path,
                        'val_acc': val_acc
                    })
                    print(f"Found {model_type} with validation accuracy: {val_acc:.2f}%")
                else:
                    print(f"Warning: No validation accuracy found in {model_path}")
            except Exception as e:
                print(f"Error loading {model_path}: {e}")
    
    # Sort by validation accuracy (descending)
    models_info.sort(key=lambda x: x['val_acc'], reverse=True)
    
    return models_info

def ensemble_predict(models_info, test_dir, output_csv, num_classes):
    """Generate predictions using an ensemble of models weighted by their validation accuracies."""
    if not models_info:
        print("No models found for ensemble prediction")
        return
    
    # Load models
    models = []
    weights = []
    # Compute softmax weights (with temprature) from validation accuracies
    accs = np.array([info['val_acc'] for info in models_info])
    accs = (accs - np.max(accs)) / (np.max(accs) - np.min(accs)) # normalize
    exp_accs = np.exp(accs/args.temp)
    softmax_weights = exp_accs / np.sum(exp_accs)
    
    for weight, info in zip(softmax_weights, models_info):
        model, _ = load_model_with_weights(info['model_type'], info['model_path'], num_classes)
        model.eval()
        models.append(model)
        weights.append(weight)
    
    print(f"Ensemble of {len(models)} models with weights: {[round(w, 3) for w in weights]}")
    
    # Create transform for test images
    transform = get_transforms()
    
    # Get all image files from test directory
    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    test_files.sort()  # Sort to ensure consistent ordering
    
    predictions = []
    
    # Make predictions for each test image
    with torch.no_grad():
        for idx, filename in enumerate(tqdm(test_files, desc="Generating predictions")):
            img_path = os.path.join(test_dir, filename)
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            # Get weighted predictions from all models
            ensemble_output = torch.zeros(1, num_classes).to(device)
            
            for model, weight in zip(models, weights):
                output = model(img_tensor)
                ensemble_output += weight * torch.softmax(output, dim=1)
            
            # Get final prediction
            pred_class = ensemble_output.argmax(dim=1).item()
            
            # Convert from 0-indexed to 1-indexed
            region_id = pred_class + 1
            
            predictions.append({
                'id': idx,
                'filename': filename,
                'Region_ID': int(region_id)
            })
    
    # Create DataFrame and save to CSV
    prediction_df = pd.DataFrame(predictions)
    
    # # Save only id and Region_ID columns for the final submission
    # submission_df = prediction_df[['id', 'Region_ID']]
    # submission_df.to_csv(output_csv, index=False)
    
    # Save a more detailed version with filenames for debugging
    detail_csv = output_csv.replace('.csv', '_with_filenames.csv')
    prediction_df.to_csv(detail_csv, index=False)
    
    print(f"Saved predictions to {output_csv} with {len(prediction_df)} entries")
    print(f"Saved detailed predictions to {detail_csv}")
    
    # # Save model info for reference
    # models_info_json = output_csv.replace('.csv', '_models_info.json')
    # with open(models_info_json, 'w') as f:
    #     json.dump(models_info, f, indent=4)
    
    # print(f"Saved models info to {models_info_json}")

def main():
    print("=== Ensemble Prediction for Region Classification ===")
    
    # Find trained models and their validation accuracies
    models_info = find_trained_models(args.models_dir)
    
    if not models_info:
        print("No trained models found. Please check your models directory.")
        return
    
    print(f"Found {len(models_info)} trained models")
    
    # Generate predictions using ensemble
    ensemble_predict(models_info, args.test_dir, args.output_csv, args.num_classes)
    
    print("=== Ensemble Prediction Complete ===")

if __name__ == "__main__":
    main()