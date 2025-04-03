#!/usr/bin/env python3
"""
Test script for Apple ProRAW support
"""

import os
import sys
import argparse
from exif_extractor import EnhancedExifExtractor
from camera_extractors import get_camera_extractor

def test_apple_raw(image_path, use_gpu=False):
    """Test Apple ProRAW extraction on a specific image
    
    Args:
        image_path: Path to the Apple ProRAW image file
        use_gpu: Whether to use GPU acceleration
    """
    print(f"\nTesting Apple ProRAW support with image: {image_path}")
    
    # Check if file exists
    if not os.path.isfile(image_path):
        print(f"Error: File not found: {image_path}")
        return
    
    # Get file extension
    file_ext = os.path.splitext(image_path)[1].lower()
    
    # Create extractor with resource management settings for Apple Silicon
    extractor = EnhancedExifExtractor(
        use_gpu=use_gpu,       # Use GPU acceleration if available
        cpu_cores=None,        # Auto-detect (n-2 on Apple Silicon)
        memory_limit_percent=75  # Cap memory usage at 75% of available RAM
    )
    
    # Extract EXIF data
    result = extractor.extract_exif(image_path)
    
    if result:
        print("\n=== Apple ProRAW Extraction Results ===")
        
        # Check if it was detected as an Apple ProRAW file
        if result.get('file_type') == 'PRORAW':
            print("✅ Successfully detected as Apple ProRAW")
        else:
            print(f"❌ Not detected as Apple ProRAW (detected as {result.get('file_type')})")
        
        # Check for Apple-specific fields
        apple_fields = [key for key in result.keys() if key.startswith('apple_')]
        if apple_fields:
            print(f"✅ Found {len(apple_fields)} Apple-specific fields")
            
            # Print Apple-specific fields
            print("\nApple ProRAW Metadata:")
            for field in sorted(apple_fields):
                print(f"  {field}: {result[field]}")
        else:
            print("❌ No Apple-specific fields found")
        
        # Check for computational photography features
        comp_photo_fields = [key for key in apple_fields if 'comp_' in key or 'hdr' in key or 'fusion' in key]
        if comp_photo_fields:
            print(f"\n✅ Found {len(comp_photo_fields)} computational photography fields")
            print("\nComputational Photography Features:")
            for field in sorted(comp_photo_fields):
                print(f"  {field}: {result[field]}")
        else:
            print("\n❓ No computational photography fields found")
        
        # Check for CoreImage processing
        ci_fields = [key for key in apple_fields if 'ci_' in key]
        if ci_fields:
            print(f"\n✅ Successfully processed with CoreImage API ({len(ci_fields)} fields)")
        else:
            print("\n❓ No CoreImage processing results found")
        
        # Print basic EXIF data
        print("\nBasic EXIF Data:")
        basic_fields = ['camera_make', 'camera_model', 'date_taken', 'exposure_time', 
                        'f_number', 'iso', 'focal_length', 'width', 'height']
        for field in basic_fields:
            if field in result:
                print(f"  {field}: {result[field]}")
    else:
        print("Failed to extract EXIF data")

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Apple ProRAW Test')
    parser.add_argument('image_path', help='Path to the Apple ProRAW image file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration if available')
    
    args = parser.parse_args()
    
    # Test Apple ProRAW support
    test_apple_raw(args.image_path, args.gpu)

if __name__ == "__main__":
    main()
