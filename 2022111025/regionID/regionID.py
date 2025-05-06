#!/usr/bin/env python3
"""
Region-Based Image Classification Model

This script implements an end-to-end solution for classifying images into 15 geographic regions
using transfer learning with lightweight CNN architectures.
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
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torchvision.datasets import ImageFolder

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create argument parser
parser = argparse.ArgumentParser(description='Train a region classifier model')
parser.add_argument('--data_dir', type=str, default='RegionIDdataset', help='Path to dataset')
parser.add_argument('--model_type', type=str, default='efficientnet_b0', 
                    choices=['efficientnet_b0', 'efficientnet_b1', 'mobilenet_v3_large', 'mobilenetv3_small', 'resnet18', 'resnet50','convnext_small', 'convnext_base', 'vit_b_16'],
                    help='Model architecture to use')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--epochs_head', type=int, default=10, help='Number of epochs for training the head')
parser.add_argument('--epochs_finetune', type=int, default=20, help='Number of epochs for fine-tuning')
parser.add_argument('--lr_head', type=float, default=3e-3, help='Learning rate for training the head')
parser.add_argument('--lr_finetune', type=float, default=1e-3, help='Learning rate for fine-tuning')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
parser.add_argument('--patience', type=int, default=7, help='Patience for early stopping')
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
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
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
    # Load pretrained model
    if model_type == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        # Replace classifier head
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
    elif model_type == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        num_ftrs = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
    elif model_type == 'mobilenetv3_small':
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        num_ftrs = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
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
    elif model_type == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
    elif model_type == 'convnext_small':
        model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, num_classes)
    elif model_type == 'convnext_base':
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, num_classes)
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
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model.to(device)

def freeze_backbone(model, model_type):
    """Freeze the backbone layers of the model."""
    if model_type.startswith('efficientnet'):
        for param in model.features.parameters():
            param.requires_grad = False
    elif model_type.startswith('mobilenet'):
        for param in model.features.parameters():
            param.requires_grad = False
    elif model_type.startswith('resnet'):
        for param in model.parameters():  
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    elif model_type.startswith('convnext'):
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif model_type.startswith('vit'):
        for param in model.parameters():
            param.requires_grad = False
        for param in model.heads.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Freezing not implemented for model type: {model_type}")
    
    return model

def unfreeze_last_block(model, model_type):
    """Unfreeze the last block of the backbone."""
    if model_type.startswith('efficientnet'):
        # Unfreeze the last block (usually 7th or 8th block)
        for i, block in enumerate(model.features):
            if i >= len(model.features) - 1:  # Last block
                for param in block.parameters():
                    param.requires_grad = True
    elif model_type.startswith('mobilenet'):
        # Unfreeze the last few layers
        layers = list(model.features.children())
        for layer in layers[-3:]:  # Last 3 layers
            for param in layer.parameters():
                param.requires_grad = True
    elif model_type.startswith('resnet'):
        # Unfreeze the last block
        for param in model.layer4.parameters():
            param.requires_grad = True
    elif model_type.startswith('convnext'):
        # Unfreeze the last block
        for param in model.features[-1].parameters():
            param.requires_grad = True
    elif model_type.startswith('vit'):
        for param in model.encoder.layers[-1].parameters():
            param.requires_grad = True
        for param in model.encoder.ln.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unfreezing not implemented for model type: {model_type}")
    
    return model

def unfreeze_all(model):
    """Unfreeze all layers of the model."""
    for param in model.parameters():
        param.requires_grad = True
    return model

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
    
    # Stage 1: Train only the head
    print("\n=== Stage 1: Training only the classification head ===")

    # Load previous model if exists
    if os.path.exists(os.path.join(args.output_dir, 'best_model_head.pth')):
        checkpoint = torch.load(os.path.join(args.output_dir, 'best_model_head.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded previous model with validation accuracy: {checkpoint['val_acc']:.2f}%")
    else:
        print("No previous model found. Starting training from scratch.")

    model = freeze_backbone(model, args.model_type)
    print(f"Frozen backbone. Now {count_trainable_parameters(model)} trainable parameters")
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=args.lr_head, weight_decay=args.weight_decay)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Training variables
    best_val_acc = 0.0
    early_stop_counter = 0
    train_losses_head, val_losses_head = [], []
    train_accs_head, val_accs_head = [], []
    
    # Train the head
    for epoch in range(args.epochs_head):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion)
        
        # Save statistics
        train_losses_head.append(train_loss)
        val_losses_head.append(val_loss)
        train_accs_head.append(train_acc)
        val_accs_head.append(val_acc)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{args.epochs_head} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save checkpoint if best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, epoch, optimizer, val_acc, 
                            os.path.join(args.output_dir, 'best_model_head.pth'))
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        # Early stopping
        if early_stop_counter >= args.patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    # Plot training head curves
    plot_training_curves(
        train_losses_head, val_losses_head, 
        train_accs_head, val_accs_head,
        os.path.join(args.output_dir, 'training_curves_head.png')
    )
    
    # Load best model from head training
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model_head.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from head training with validation accuracy: {checkpoint['val_acc']:.2f}%")

    # Load previous model if exists and better accuracy
    if os.path.exists(os.path.join(args.output_dir, 'best_model_finetune.pth')):
        checkpoint = torch.load(os.path.join(args.output_dir, 'best_model_finetune.pth'))
        if checkpoint['val_acc'] > best_val_acc:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded previous model with validation accuracy: {checkpoint['val_acc']:.2f}%")
    else:
        print("No previous model found for fine-tuning. Starting fine-tuning from the head model.")
    
    # Stage 2: Fine-tune
    print("\n=== Stage 2: Fine-tuning the model ===")
    
    # First unfreeze the last block
    model = unfreeze_last_block(model, args.model_type)
    print(f"Unfrozen last block. Now {count_trainable_parameters(model)} trainable parameters")
    
    # New optimizer with lower learning rate
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=args.lr_finetune, weight_decay=args.weight_decay)
    
    # Cosine annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs_finetune)
    
    # Training variables
    best_val_acc = 0.0
    early_stop_counter = 0
    train_losses_ft, val_losses_ft = [], []
    train_accs_ft, val_accs_ft = [], []
    early_unfreeze = -1
    
    # Fine-tune the model
    for epoch in range(args.epochs_finetune):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion)
        
        # Save statistics
        train_losses_ft.append(train_loss)
        val_losses_ft.append(val_loss)
        train_accs_ft.append(train_acc)
        val_accs_ft.append(val_acc)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{args.epochs_finetune} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint if best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, epoch, optimizer, val_acc, 
                            os.path.join(args.output_dir, 'best_model_finetune.pth'))
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        # Early stopping
        if early_stop_counter >= args.patience:
            if epoch > args.epochs_finetune // 2 and early_unfreeze != -1:
                print(f"Early stopping after {epoch+1} epochs")
                break
            else:
                early_unfreeze = epoch + 1
                model = unfreeze_all(model)
                print(f"Model reached early stopping criteria. Unfreezing all layers.")
                print(f"Unfrozen all layers. Now {count_trainable_parameters(model)} trainable parameters")
                # Reduce learning rate further for full model fine-tuning
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr_finetune / 10
                # Reset early stopping counter
                early_stop_counter = 0
                # Continue training with full model fine-tuning
                print(f"Continuing training with full model fine-tuning.")
                continue
        
        # After half of the fine-tuning epochs, unfreeze all layers for full fine-tuning
        if epoch == args.epochs_finetune // 2 and early_unfreeze == -1:
            early_unfreeze = epoch + 1
            model = unfreeze_all(model)
            print(f"Unfrozen all layers. Now {count_trainable_parameters(model)} trainable parameters")
            # Reduce learning rate further for full model fine-tuning
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr_finetune / 10
    
    print(f"Finished fine-tuning after {epoch+1} epochs")
    print(f"Early unfreeze epoch: {early_unfreeze}")

    # Plot fine-tuning curves
    plot_training_curves(
        train_losses_ft, val_losses_ft, 
        train_accs_ft, val_accs_ft,
        os.path.join(args.output_dir, 'training_curves_finetune.png')
    )
    
    # Load best fine-tuned model
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model_finetune.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best fine-tuned model with validation accuracy: {checkpoint['val_acc']:.2f}%")
    
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