#!/usr/bin/env python3
"""
Convert JPG image to .ubyte format for MNIST inference.

Usage:
    python convert_jpg_to_ubyte.py input.jpg output.ubyte

This script performs the following steps:
    1. Loads the input JPG image using PIL/Pillow.
    2. Converts the image to grayscale if it is not already.
    3. Resizes the image to 28x28 pixels (the MNIST standard size).
    4. Inverts the image colors if the background is white (MNIST expects white digits on a black background).
    5. Normalizes pixel values to the [0, 1] float range.
    6. Saves the processed image as a binary .ubyte file in float32 format, suitable for C++ inference code.
    7. Displays an ASCII art preview of the processed image for quick visual verification.
"""

import numpy as np
import sys
from PIL import Image
import os

def convert_jpg_to_ubyte(jpg_path, ubyte_path):
    """
    Convert a JPG image to MNIST-compatible .ubyte format
    
    Args:
        jpg_path: Path to input JPG image
        ubyte_path: Path to output .ubyte file
    """
    
    # Load the image
    try:
        img = Image.open(jpg_path)
        print(f"Loaded image: {img.size}, mode: {img.mode}")
    except Exception as e:
        print(f"Error loading image {jpg_path}: {e}")
        return False
    
    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')
        print("Converted to grayscale")
    
    # Resize to 28x28 (MNIST standard)
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    print("Resized to 28x28")
    
    # Convert to numpy array
    img_array = np.array(img, dtype=np.uint8)
    
    # MNIST images are white digits on black background
    # If your image has black digits on white background, invert it
    # Check if we need to invert (assume we need to if background is bright)
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
        print("Inverted image (black digits on white -> white digits on black)")
    
    # Normalize to [0, 1] range and convert to float32
    img_normalized = img_array.astype(np.float32) / 255.0
    
    # Flatten to 1D array (784 elements)
    img_flat = img_normalized.flatten()
    
    # Save as .ubyte (raw binary format)
    try:
        # Save as float32 binary data
        with open(ubyte_path, 'wb') as f:
            img_flat.astype(np.float32).tobytes()
            f.write(img_flat.astype(np.float32).tobytes())
        
        print(f"Saved to {ubyte_path}")
        print(f"Image stats: min={img_flat.min():.3f}, max={img_flat.max():.3f}, mean={img_flat.mean():.3f}")
        
        return True
        
    except Exception as e:
        print(f"Error saving to {ubyte_path}: {e}")
        return False

def preview_image(img_array, title="Image Preview"):
    """
    Print a simple ASCII preview of the 28x28 image
    """
    print(f"\n{title}:")
    print("+" + "-" * 28 + "+")
    
    for row in img_array:
        line = "|"
        for pixel in row:
            if pixel > 0.5:
                line += "#"
            elif pixel > 0.2:
                line += "."
            else:
                line += " "
        line += "|"
        print(line)
    
    print("+" + "-" * 28 + "+")

def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_jpg_to_ubyte.py input.jpg output.ubyte")
        print("Example: python convert_jpg_to_ubyte.py img_5.jpg image_5.ubyte")
        sys.exit(1)
    
    jpg_path = sys.argv[1]
    ubyte_path = sys.argv[2]
    
    # Check if input file exists
    if not os.path.exists(jpg_path):
        print(f"Error: Input file {jpg_path} does not exist")
        sys.exit(1)
    
    print(f"Converting {jpg_path} to {ubyte_path}")
    
    # Perform conversion
    success = convert_jpg_to_ubyte(jpg_path, ubyte_path)
    
    if success:
        print(f"\nConversion successful!")
        
        # Load and preview the result
        try:
            with open(ubyte_path, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.float32)
                img_2d = data.reshape(28, 28)
                preview_image(img_2d, "Converted Image Preview")
        except Exception as e:
            print(f"Could not preview result: {e}")
        
        print(f"\nNow you can run inference with:")
        print(f"./src/inference_engine ../models/mnist_ffn.onnx ../{ubyte_path}")
    else:
        print("Conversion failed")
        sys.exit(1)

if __name__ == "__main__":
    main()