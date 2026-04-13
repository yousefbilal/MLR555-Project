#!/bin/bash

ORL_URL="https://www.kaggle.com/api/v1/datasets/download/tavarez/the-orl-database-for-training-and-testing"
COIL20_URL="https://www.kaggle.com/api/v1/datasets/download/cyx6666/coil20"

BASE_DIR="data"
mkdir -p "$BASE_DIR"

# Function to download and extract
download_and_extract() {
    local url=$1
    local folder_name=$2
    local target_dir="$BASE_DIR/$folder_name"
    local zip_file="$BASE_DIR/${folder_name}.zip"

    echo "Fetching $folder_name..."
    mkdir -p "$target_dir"
    
    curl -L "$url" -o "$zip_file"

    echo "Extracting into $target_dir..."
    unzip -q -o "$zip_file" -d "$target_dir"
    echo "Cleaning up..."
    rm "$zip_file"
    echo "--------------------------"
}

echo "Downloading all datasets to the '$BASE_DIR' directory..."
echo "=========================================="

download_and_extract "$ORL_URL" "orl"
