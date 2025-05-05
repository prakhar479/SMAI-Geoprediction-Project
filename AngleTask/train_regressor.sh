#!/bin/bash
# Script to run the angle regressor training with different model architectures
EPOCHS=50
EMBEDDING_DIM=32


# Create output directories
mkdir -p outputs/resnet18
mkdir -p outputs/resnet50
mkdir -p outputs/efficientnet_b0
mkdir -p outputs/efficientnet_b3
mkdir -p outputs/mobilenet_v2
mkdir -p outputs/mobilenet_v3_large
mkdir -p outputs/convnext_small
mkdir -p outputs/convnext_base

# Train with EfficientNet-B0 
echo "Training with EfficientNet-B0"
python angle.py --backbone efficientnet_b0 --output_dir outputs/efficientnet_b0 --epochs $EPOCHS --embed_dim $EMBEDDING_DIM --pretrained
# Train with EfficientNet-B3 
echo "Training with EfficientNet-B3"
python angle.py --backbone efficientnet_b3 --output_dir outputs/efficientnet_b3 --epochs $EPOCHS --embed_dim $EMBEDDING_DIM --pretrained

# Train with ResNet18 
echo "Training with ResNet18"
python angle.py --backbone resnet18 --output_dir outputs/resnet18 --epochs $EPOCHS --embed_dim $EMBEDDING_DIM --pretrained
# Train with ResNet50
echo "Training with ResNet50"
python angle.py --backbone resnet50 --output_dir outputs/resnet50 --epochs $EPOCHS --embed_dim $EMBEDDING_DIM --pretrained

# Train with MobileNetV2
echo "Training with MobileNetV2"
python angle.py --backbone mobilenet_v2 --output_dir outputs/mobilenet_v2 --epochs $EPOCHS --embed_dim $EMBEDDING_DIM --pretrained
# Train with MobileNetV3
echo "Training with MobileNetV3"
python angle.py --backbone mobilenet_v3_large --output_dir outputs/mobilenet_v3_large --epochs $EPOCHS --embed_dim $EMBEDDING_DIM --pretrained

# Train with ConvNext-Small
echo "Training with ConvNext-Small"
python angle.py --backbone convnext_small --output_dir outputs/convnext_small --epochs $EPOCHS --embed_dim $EMBEDDING_DIM --pretrained
# Train with ConvNext-Base
echo "Training with ConvNext-Base"
python angle.py --backbone convnext_base --output_dir outputs/convnext_base --epochs $EPOCHS --embed_dim $EMBEDDING_DIM --pretrained

echo "All training completed!"