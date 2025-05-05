#!/bin/bash
# Script to run the region classifier training with different model architectures
HEAD_EPOCHS=30
FINETUNE_EPOCHS=50


# Create output directories
mkdir -p outputs/efficientnet_b0
mkdir -p outputs/efficientnet_b1
mkdir -p outputs/mobilenet_v3_large
mkdir -p outputs/resnet18
mkdir -p outputs/resnet50
mkdir -p outputs/convnext_small
mkdir -p outputs/convnext_base
mkdir -p outputs/vit_b_16

# Train with EfficientNet-B0 (default parameters)
echo "Training with EfficientNet-B0"
python regionID.py --model_type efficientnet_b0 --output_dir outputs/efficientnet_b0 --epochs_head $HEAD_EPOCHS --epochs_finetune $FINETUNE_EPOCHS

# Train with EfficientNet-B1
echo "Training with EfficientNet-B1"
python regionID.py --model_type efficientnet_b1 --output_dir outputs/efficientnet_b1 --epochs_head $HEAD_EPOCHS --epochs_finetune $FINETUNE_EPOCHS

# Train with MobileNetV3-Large
echo "Training with MobileNetV3-Large"
python regionID.py --model_type mobilenet_v3_large --output_dir outputs/mobilenet_v3_large --epochs_head $HEAD_EPOCHS --epochs_finetune $FINETUNE_EPOCHS

# Train with ResNet-18
echo "Training with ResNet-18"
python regionID.py --model_type resnet18 --output_dir outputs/resnet18 --epochs_head $HEAD_EPOCHS --epochs_finetune $FINETUNE_EPOCHS

# Train with ResNet-50
echo "Training with ResNet-50"
python regionID.py --model_type resnet50 --output_dir outputs/resnet50 --epochs_head $HEAD_EPOCHS --epochs_finetune $FINETUNE_EPOCHS

# Train with Convnext-Small
echo "Training with Convnext-Small"
python regionID.py --model_type convnext_small --output_dir outputs/convnext_small --epochs_head $HEAD_EPOCHS --epochs_finetune $FINETUNE_EPOCHS

# Train with Convnext-Base
echo "Training with Convnext-Base"
python regionID.py --model_type convnext_base --output_dir outputs/convnext_base --epochs_head $HEAD_EPOCHS --epochs_finetune $FINETUNE_EPOCHS

# Train with VIT_B_16
echo "Training with VIT B 16"
python regionID.py --model_type vit_b_16 --output_dir outputs/vit_b_16 --epochs_head $HEAD_EPOCHS --epochs_finetune $FINETUNE_EPOCHS

echo "All training completed!"