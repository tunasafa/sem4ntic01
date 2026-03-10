#!/usr/bin/env python3
"""
Download pre-trained YOLOv8 segmentation models.

This script downloads the required models from Ultralytics' CDN.
Run this after installing requirements.txt if models are not present.

Usage:
    python download_models.py

Models will be downloaded to: mini-services/segmentation-service/
"""

import os
from pathlib import Path

try:
    from ultralytics import YOLO
    print("✓ ultralytics is installed")
except ImportError:
    print("✗ ultralytics not installed. Please run:")
    print("  pip install -r requirements.txt")
    exit(1)

# Models to download (choose based on your needs)
# nano: Fastest, smallest accuracy
# small: Good balance
# medium: Better accuracy, slower
# large: Best accuracy, slowest
# xlarge: Maximum accuracy, very slow

MODELS = {
    "yolov8n-seg.pt": "Nano - Fastest (recommended for testing)",
    "yolov8s-seg.pt": "Small - Balanced (recommended for production)",
    # "yolov8m-seg.pt": "Medium - Better accuracy",
    # "yolov8l-seg.pt": "Large - Best accuracy",
    # "yolov8x-seg.pt": "XLarge - Maximum accuracy",
}

def download_model(model_name: str):
    """Download a single YOLO model."""
    model_path = Path(model_name)
    
    if model_path.exists():
        print(f"✓ {model_name} already exists, skipping...")
        return
    
    print(f"Downloading {model_name}...")
    try:
        # This will download the model automatically
        model = YOLO(model_name)
        print(f"✓ {model_name} downloaded successfully")
    except Exception as e:
        print(f"✗ Failed to download {model_name}: {e}")

def main():
    print("=" * 60)
    print("YOLOv8 Segmentation Model Downloader")
    print("=" * 60)
    print()
    
    # Create models directory if it doesn't exist
    models_dir = Path(__file__).parent
    os.chdir(models_dir)
    
    print(f"Downloading models to: {models_dir}")
    print()
    
    # Download each model
    for model_name, description in MODELS.items():
        print(f"[{description}]")
        download_model(model_name)
        print()
    
    print("=" * 60)
    print("Model download complete!")
    print()
    print("Available models:")
    for model_name in MODELS.keys():
        exists = "✓" if Path(model_name).exists() else "✗"
        print(f"  {exists} {model_name}")
    
    print()
    print("To use a different model, edit the MODELS dictionary in this script.")
    print("=" * 60)

if __name__ == "__main__":
    main()
