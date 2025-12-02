import numpy as np
import cv2
import os
from pathlib import Path

def convert_npy_to_images(npy_dir="data_npy", output_dir="fruit_images", max_images_per_class=10000):
    """
    Convert numpy bitmap files (.npy) from Quick Draw dataset to PNG images.

    Args:
        npy_dir: Directory containing .npy files
        output_dir: Directory where PNG images will be saved
        max_images_per_class: Maximum number of images to convert per fruit class
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # List of fruit .npy files to process
    fruit_files = [
        "full_numpy_bitmap_apple.npy",
        "full_numpy_bitmap_banana.npy",
        "full_numpy_bitmap_blueberry.npy",
        "full_numpy_bitmap_grapes.npy",
        "full_numpy_bitmap_pineapple.npy",
        "full_numpy_bitmap_strawberry.npy",
        "full_numpy_bitmap_watermelon.npy"
    ]

    total_converted = 0

    for fruit_file in fruit_files:
        # Extract fruit name from filename
        fruit_name = fruit_file.replace("full_numpy_bitmap_", "").replace(".npy", "")

        # Create subdirectory for this fruit
        fruit_output_dir = os.path.join(output_dir, fruit_name)
        os.makedirs(fruit_output_dir, exist_ok=True)

        # Full path to .npy file
        npy_path = os.path.join(npy_dir, fruit_file)

        # Check if file exists
        if not os.path.exists(npy_path):
            print(f"Warning: File not found - {npy_path}")
            continue

        print(f"\nProcessing {fruit_name}...")
        print(f"Loading {npy_path}...")

        try:
            # Load the numpy array
            # Quick Draw numpy bitmaps are stored as (N, 784) arrays
            # where N is number of images and 784 = 28x28 pixels
            data = np.load(npy_path)

            # Limit number of images
            num_images = min(len(data), max_images_per_class)

            print(f"Converting {num_images} images for {fruit_name}...")

            # Convert each image
            for i in range(num_images):
                # Reshape from (784,) to (28, 28)
                img = data[i].reshape(28, 28)

                # Convert to uint8 (Quick Draw images are already 0-255)
                img = img.astype(np.uint8)

                # Save as PNG
                output_path = os.path.join(fruit_output_dir, f"{fruit_name}_{i:05d}.png")
                cv2.imwrite(output_path, img)

                # Progress indicator
                if (i + 1) % 1000 == 0:
                    print(f"  Converted {i + 1}/{num_images} images...")

            print(f"✓ Completed {fruit_name}: {num_images} images saved to {fruit_output_dir}")
            total_converted += num_images

        except Exception as e:
            print(f"Error processing {fruit_file}: {str(e)}")
            continue

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"Total images converted: {total_converted}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Run the conversion
    convert_npy_to_images(
        npy_dir="data_npy",
        output_dir="fruit_images",
        max_images_per_class=10000
    )
