#!/usr/bin/env python3
"""
Test script for all camera extractors
Tests extraction of camera-specific metadata from various RAW formats
"""

import os
import sys
import json
from typing import Dict, Any, List
from exif_extractor import EnhancedExifExtractor
import multiprocessing

def test_file(image_path: str, use_gpu: bool = False) -> Dict[str, Any]:
    """
    Test extraction on a specific image
    
    Args:
        image_path: Path to the image
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Dictionary with extracted metadata
    """
    print(f"\n{'='*80}\nTesting extraction on: {image_path}")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return {}
    
    # Get file extension
    _, ext = os.path.splitext(image_path)
    
    # Create extractor with Apple Silicon optimized settings
    # Use n-2 cores where n is total cores
    total_cores = multiprocessing.cpu_count()
    cpu_cores = max(1, total_cores - 2)
    
    # Cap memory at 75% of available RAM
    memory_limit_percent = 75
    
    print(f"Using {cpu_cores}/{total_cores} CPU cores, {memory_limit_percent}% memory limit, GPU: {use_gpu}")
    
    extractor = EnhancedExifExtractor(
        use_gpu=use_gpu,
        memory_limit_percent=memory_limit_percent,
        cpu_cores=cpu_cores
    )
    
    # Extract EXIF data
    exif_data = extractor.extract_exif(image_path)
    
    # Print basic info
    print(f"File: {os.path.basename(image_path)} ({ext})")
    print(f"Camera Make: {exif_data.get('camera_make', 'Unknown')}")
    print(f"Camera Model: {exif_data.get('camera_model', 'Unknown')}")
    
    # Check for camera-specific data
    camera_make = exif_data.get('camera_make', '').upper()
    
    # Find camera-specific fields
    prefixes = {
        'SONY': 'sony_',
        'NIKON': 'nikon_',
        'CANON': 'canon_',
        'LEICA': 'leica_',
        'FUJI': 'fuji_',
        'FUJIFILM': 'fuji_',
        'APPLE': 'apple_',
        'PANASONIC': 'panasonic_'
    }
    
    # Get the prefix for this camera
    prefix = None
    for make, p in prefixes.items():
        if make in camera_make:
            prefix = p
            break
    
    if not prefix and 'DNG' in ext.upper():
        prefix = 'dng_'
    
    # Count camera-specific fields
    if prefix:
        specific_fields = [key for key in exif_data.keys() if key.startswith(prefix)]
        print(f"Found {len(specific_fields)} {prefix.strip('_')}-specific fields")
        
        # Print some examples
        if specific_fields:
            print(f"\nExample {prefix.strip('_')}-specific fields:")
            for field in sorted(specific_fields)[:10]:  # Show first 10 fields
                print(f"  {field}: {exif_data[field]}")
            
            if len(specific_fields) > 10:
                print(f"  ... and {len(specific_fields) - 10} more fields")
    else:
        print("No camera-specific fields found")
    
    return exif_data

def main() -> None:
    """Main function"""
    # Set test directory
    test_dir = "Test_for_EXIF"
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found: {test_dir}")
        return
    
    # Get GPU setting from command line
    use_gpu = "--gpu" in sys.argv
    
    # Test files for each camera type
    test_files = [
        os.path.join(test_dir, "puna teal - flapping.ARW"),  # Sony
        os.path.join(test_dir, "_DSC2894.NEF"),              # Nikon
        os.path.join(test_dir, "8P4A3648.CR3"),              # Canon
        os.path.join(test_dir, "L1000699.DNG"),              # Leica DNG
        os.path.join(test_dir, "DSCF2803.RAF"),              # Fujifilm
        os.path.join(test_dir, "IMG_3341.DNG"),              # Apple ProRAW
        os.path.join(test_dir, "RAW_PANASONIC_DMC-GH4.RW2"), # Panasonic
    ]
    
    # Process each file
    results = {}
    for file_path in test_files:
        if os.path.exists(file_path):
            results[os.path.basename(file_path)] = test_file(file_path, use_gpu)
        else:
            print(f"Skipping missing file: {file_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for file_name, data in results.items():
        camera = f"{data.get('camera_make', 'Unknown')} {data.get('camera_model', 'Unknown')}"
        print(f"{file_name}: {camera} - {len(data)} fields extracted")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
