#!/usr/bin/env python3
"""
Region-Based Image Classification Model

This script implements an end-to-end solution for classifying images into geographic regions
using transfer learning with various CNN architectures.
"""

import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.datasets import ImageFolder

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create argument parser
parser = argparse.ArgumentParser(description='Train a region classifier model')
parser.add_argument('--data_dir', type=str, default='RegionIDDataset', help='Path to dataset')
parser.add_argument('--model_type', type=str, default='efficientnet_b0', 
                    choices=['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
                             'mobilenet_v3_large', 'mobilenet_v3_small', 
                             'resnet18', 'resnet34', 'resnet50', 'resnet101',
                             'convnext_tiny', 'convnext_small', 'convnext_base', 
                             'vit_b_16', 'vit_b_32', 'vit_l_16',
                             'swin_t', 'swin_s', 'swin_b'],
                    help='Model architecture to use')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs for training')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save outputs')
parser.add_argument('--num_workers', type=int, default=os.cpu_count() - 1, help='Number of workers for data loading')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()

# Set random seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Define data transformations
def get_transforms():
    """Create data transformations for train and validation sets."""
    # ImageNet normalization values
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    return train_transform, val_transform

def create_data_loaders():
    """Create data loaders for training and validation."""
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = ImageFolder(
        os.path.join(args.data_dir, 'train'),
        transform=train_transform
    )
    
    val_dataset = ImageFolder(
        os.path.join(args.data_dir, 'val'),
        transform=val_transform
    )
    
    # Print dataset information
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Class mapping: {train_dataset.class_to_idx}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, train_dataset.classes

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

def count_trainable_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]") as pbar:
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix(loss=loss.item(), acc=f"{100 * correct / total:.2f}%")
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion):
    """Validate the model on the validation set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        with tqdm(val_loader, desc="Validation") as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # For confusion matrix
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix(loss=loss.item(), acc=f"{100 * correct / total:.2f}%")
    
    val_loss = running_loss / len(val_loader.dataset)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc, all_preds, all_labels

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', save_path=None):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    plt.close()

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """Plot training and validation curves."""
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training curves saved to {save_path}")
    plt.close()

def save_checkpoint(model, epoch, optimizer, val_acc, filename):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }, filename)
    print(f"Checkpoint saved to {filename}")

def main():
    start_time = time.time()
    
    # Create data loaders
    train_loader, val_loader, class_names = create_data_loaders()
    num_classes = len(class_names)
    
    # Create model
    model = create_model(args.model_type, num_classes)
    print(f"Created {args.model_type} model with {count_trainable_parameters(model)} trainable parameters")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Cosine annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Check if previous model exists
    if os.path.exists(os.path.join(args.output_dir, 'best_model.pth')):
        checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded previous model with validation accuracy: {checkpoint['val_acc']:.2f}%")
    else:
        print("No previous model found. Starting training from scratch.")
    
    # Training variables
    best_val_acc = 0.0
    early_stop_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    # Train the model
    print(f"\n=== Training {args.model_type} model ===")
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion)
        
        # Save statistics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint if best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, epoch, optimizer, val_acc, 
                            os.path.join(args.output_dir, 'best_model.pth'))
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        # Early stopping
        if early_stop_counter >= args.patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    # Plot training curves
    plot_training_curves(
        train_losses, val_losses, 
        train_accs, val_accs,
        os.path.join(args.output_dir, 'training_curves.png')
    )
    
    # Load best model
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model with validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    val_loss, val_acc, all_preds, all_labels = validate(model, val_loader, criterion)
    print(f"Final validation accuracy: {val_acc:.2f}%")
    
    # Compute and plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(
        cm, class_names=class_names,
        title=f'Confusion Matrix - {args.model_type} (Acc: {val_acc:.2f}%)',
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Save classification report to file
    with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Model: {args.model_type}\n")
        f.write(f"Final validation accuracy: {val_acc:.2f}%\n\n")
        f.write(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
    
    # Print execution time
    elapsed_time = time.time() - start_time
    print(f"\nExecution time: {elapsed_time//60:.0f}m {elapsed_time%60:.0f}s")

if __name__ == "__main__":
    main()