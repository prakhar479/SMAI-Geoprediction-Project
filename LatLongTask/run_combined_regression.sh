#!/bin/bash
# Script to run the angle regressor training with different model architectures
EPOCHS=100
EMBEDDING_DIM=64


# Create output directories
mkdir -p outputs/resnet34
mkdir -p outputs/efficientnet_b0
mkdir -p outputs/mobilenet_v3_small
mkdir -p outputs/vit_b_16
mkdir -p outputs/swin_b


# Train resnet34
echo "Training with Resnet34"
python latlong.py --backbone resnet34 --epochs $EPOCHS --embedding_dim $EMBEDDING_DIM --output_dir outputs/resnet34

# Train efficientnet_b0
echo "Training with EfficientnetB0"
python latlong.py --backbone efficientnet_b0 --epochs $EPOCHS --embedding_dim $EMBEDDING_DIM --output_dir outputs/efficientnet_b0

# Train mobilenet_v3_small
echo "Training with MobilenetV3Small"
python latlong.py --backbone mobilenet_v3_small --epochs $EPOCHS --embedding_dim $EMBEDDING_DIM --output_dir outputs/mobilenet_v3_small

# Train vit_b_16
echo "Training with VIT_B_16"
python latlong.py --backbone vit_b_16 --epochs $EPOCHS --embedding_dim $EMBEDDING_DIM --output_dir outputs/vit_b_16

# Train swin_b
echo "Training with SWIN B"
python latlong.py --backbone swin_b --epochs $EPOCHS --embedding_dim $EMBEDDING_DIM --output_dir outputs/swin_b