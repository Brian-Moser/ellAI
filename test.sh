#!/bin/bash

# Test with fine-tuned model (vanilla saliency)
echo "Testing with fine-tuned model (vanilla saliency)..."
python main.py --dataset_path datasets/leaves_healthy \
               --mode test \
               --image_path test_images/bergahorn_healthy.jpg \
               --model mobilenet_v2 \
               --use_finetuning \
               --vis vanilla \
               --output_path .

# Test with fine-tuned model (regular CAM)
echo "Testing with fine-tuned model (regular CAM)..."
python main.py --dataset_path datasets/leaves_healthy \
               --mode test \
               --image_path test_images/bergahorn_healthy.jpg \
               --model mobilenet_v2 \
               --use_finetuning \
               --vis cam \
               --output_path .

# Test with fine-tuned model (Grad-CAM++)
echo "Testing with fine-tuned model (Grad-CAM++)..."
python main.py --dataset_path datasets/leaves_healthy \
               --mode test \
               --image_path test_images/bergahorn_healthy.jpg \
               --model mobilenet_v2 \
               --use_finetuning \
               --vis gradcam_pp \
               --output_path .

# Test with KNN
echo "Testing with KNN..."
python main.py --dataset_path datasets/leaves_healthy \
               --mode test \
               --image_path test_images/bergahorn_healthy.jpg \
               --output_path .