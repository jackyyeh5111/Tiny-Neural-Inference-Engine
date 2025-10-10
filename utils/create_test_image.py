#!/usr/bin/env python3
"""
MNIST Test Image Generator

This utility creates simple test images for the MNIST inference engine.
It can create synthetic digit patterns or random noise for testing purposes.
"""

import numpy as np
import sys
import os

def create_digit_pattern(digit, filename):
    """Create a simple synthetic pattern for a given digit (0-9)"""
    image = np.zeros((28, 28), dtype=np.uint8)
    
    if digit == 0:
        # Create a circle
        center = (14, 14)
        radius = 8
        y, x = np.ogrid[:28, :28]
        mask = ((x - center[0])**2 + (y - center[1])**2 <= radius**2) & \
               ((x - center[0])**2 + (y - center[1])**2 >= (radius-3)**2)
        image[mask] = 255
        
    elif digit == 1:
        # Create a vertical line
        image[5:23, 13:15] = 255
        
    elif digit == 2:
        # Create a horizontal S-like pattern
        image[8:10, 8:20] = 255  # Top horizontal
        image[8:15, 18:20] = 255  # Right vertical
        image[13:15, 8:20] = 255  # Middle horizontal
        image[13:20, 8:10] = 255  # Left vertical
        image[18:20, 8:20] = 255  # Bottom horizontal
        
    elif digit == 3:
        # Create number 3 pattern
        image[8:10, 8:18] = 255   # Top horizontal
        image[13:15, 8:18] = 255  # Middle horizontal
        image[18:20, 8:18] = 255  # Bottom horizontal
        image[8:20, 16:18] = 255  # Right vertical
        
    elif digit == 4:
        # Create number 4 pattern
        image[8:15, 8:10] = 255   # Left vertical
        image[8:15, 16:18] = 255  # Right vertical
        image[13:15, 8:18] = 255  # Middle horizontal
        image[15:20, 16:18] = 255 # Right bottom vertical
        
    elif digit == 5:
        # Create number 5 pattern
        image[8:10, 8:18] = 255   # Top horizontal
        image[8:15, 8:10] = 255   # Left vertical top
        image[13:15, 8:18] = 255  # Middle horizontal
        image[13:20, 16:18] = 255 # Right vertical bottom
        image[18:20, 8:18] = 255  # Bottom horizontal
        
    else:
        # For digits 6-9 or invalid, create a random pattern
        image[10:18, 10:18] = 128
        
    # Flatten and save as binary file
    image_bytes = image.flatten().tobytes()
    
    with open(filename, 'wb') as f:
        f.write(image_bytes)
    
    print(f"Created test image for digit {digit}: {filename}")
    print(f"File size: {len(image_bytes)} bytes")

def create_random_image(filename):
    """Create a random test image"""
    image = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
    image_bytes = image.flatten().tobytes()
    
    with open(filename, 'wb') as f:
        f.write(image_bytes)
    
    print(f"Created random test image: {filename}")
    print(f"File size: {len(image_bytes)} bytes")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 create_test_image.py <digit>         # Create pattern for digit 0-9")
        print("  python3 create_test_image.py random          # Create random noise image")
        print("  python3 create_test_image.py all             # Create patterns for all digits 0-9")
        sys.exit(1)
    
    # Create inputs directory if it doesn't exist
    inputs_dir = "../inputs"
    os.makedirs(inputs_dir, exist_ok=True)
    
    arg = sys.argv[1].lower()
    
    if arg == "random":
        filename = os.path.join(inputs_dir, "random_test.ubyte")
        create_random_image(filename)
        
    elif arg == "all":
        for digit in range(10):
            filename = os.path.join(inputs_dir, f"digit_{digit}_test.ubyte")
            create_digit_pattern(digit, filename)
            
    elif arg.isdigit() and 0 <= int(arg) <= 9:
        digit = int(arg)
        filename = os.path.join(inputs_dir, f"digit_{digit}_test.ubyte")
        create_digit_pattern(digit, filename)
        
    else:
        print(f"Invalid argument: {arg}")
        print("Use 'random', 'all', or a digit 0-9")
        sys.exit(1)

if __name__ == "__main__":
    main()