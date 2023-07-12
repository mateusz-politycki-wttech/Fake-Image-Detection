#!/bin/bash

names=("airplane" "bicycle" "bird" "boat" "bottle" "bus" "car" "cat" "chair" "cow" "diningtable" "dog" "horse" "motorbike" "person" "pottedplant" "sheep" "sofa" "tvmonitor" "train")

# Define source and destination directories
source_dir="dataset-huge/train"
destination_dir="dataset/train"

# Iterate over each category
for name in "${names[@]}"; do
  # Copy first 200 images from source to destination
  mkdir -p "dataset/train/$name/0_real"
  mkdir -p "dataset/train/$name/1_fake"
  find "$source_dir/$name/0_real" -type f | head -n 5000 | xargs cp -t "$destination_dir/$name/0_real"
  find "$source_dir/$name/1_fake" -type f | head -n 5000 | xargs cp -t "$destination_dir/$name/1_fake"
done