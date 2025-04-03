#!/usr/bin/env python3
"""
Detailed test script for all camera extractors
Tests extraction of camera-specific metadata from various RAW formats
"""

import os
import sys
import json
from typing import Dict, Any, List
from exif_extractor import EnhancedExifExtractor
import multiprocessing
from camera_extractors import *
import time

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
    
    # First test with the general extractor
    print("\n--- Testing with EnhancedExifExtractor ---")
    start_time = time.time()
    extractor = EnhancedExifExtractor(
        use_gpu=use_gpu,
        memory_limit_percent=memory_limit_percent,
        cpu_cores=cpu_cores
    )
    
    # Extract EXIF data
    exif_data = extractor.extract_exif(image_path)
    general_time = time.time() - start_time
    
    # Print basic info
    print(f"File: {os.path.basename(image_path)} ({ext})")
    print(f"Camera Make: {exif_data.get('camera_make', 'Unknown')}")
    print(f"Camera Model: {exif_data.get('camera_model', 'Unknown')}")
    print(f"Extraction time: {general_time:.2f} seconds")
    
    # Now test with the specific camera extractor
    camera_make = exif_data.get('camera_make', '').upper()
    
    # Find the appropriate extractor
    extractor_class = None
    if 'SONY' in camera_make:
        from camera_extractors.sony_extractor import SonyExtractor
        extractor_class = SonyExtractor
        prefix = 'sony_'
    elif 'NIKON' in camera_make:
        from camera_extractors.nikon_extractor import NikonExtractor
        extractor_class = NikonExtractor
        prefix = 'nikon_'
    elif 'CANON' in camera_make:
        from camera_extractors.canon_extractor import CanonExtractor
        extractor_class = CanonExtractor
        prefix = 'canon_'
    elif 'FUJI' in camera_make:
        from camera_extractors.fujifilm_extractor import FujifilmExtractor
        extractor_class = FujifilmExtractor
        prefix = 'fuji_'
    elif 'APPLE' in camera_make or 'IPHONE' in camera_make:
        from camera_extractors.apple_raw_extractor import AppleRawExtractor
        extractor_class = AppleRawExtractor
        prefix = 'apple_'
    elif 'PANASONIC' in camera_make or 'LUMIX' in camera_make:
        from camera_extractors.panasonic_extractor import PanasonicExtractor
        extractor_class = PanasonicExtractor
        prefix = 'panasonic_'
    elif '.DNG' in ext.upper():
        from camera_extractors.dng_extractor import DngExtractor
        extractor_class = DngExtractor
        prefix = 'dng_'
    
    if extractor_class:
        print(f"\n--- Testing with {extractor_class.__name__} ---")
        start_time = time.time()
        specific_extractor = extractor_class(
            use_gpu=use_gpu,
            memory_limit=memory_limit_percent/100.0,
            cpu_cores=cpu_cores
        )
        
        # Extract metadata with the specific extractor
        specific_data = {}
        try:
            specific_data = specific_extractor.extract_metadata(image_path, exif_data)
            specific_time = time.time() - start_time
            print(f"Specific extraction time: {specific_time:.2f} seconds")
            
            # Count camera-specific fields
            specific_fields = [key for key in specific_data.keys() if key.startswith(prefix)]
            print(f"Found {len(specific_fields)} {prefix.strip('_')}-specific fields")
            
            # Print some examples
            if specific_fields:
                print(f"\nExample {prefix.strip('_')}-specific fields:")
                for field in sorted(specific_fields)[:10]:  # Show first 10 fields
                    value = specific_data[field]
                    if isinstance(value, (list, dict)) and len(str(value)) > 100:
                        value = str(value)[:100] + "..."
                    print(f"  {field}: {value}")
                
                if len(specific_fields) > 10:
                    print(f"  ... and {len(specific_fields) - 10} more fields")
                
                # Test raw processing
                print("\n--- Testing RAW processing ---")
                start_time = time.time()
                raw_data = specific_extractor.process_raw(image_path, exif_data)
                raw_time = time.time() - start_time
                print(f"RAW processing time: {raw_time:.2f} seconds")
                
                raw_fields = [key for key in raw_data.keys() if key.startswith(prefix)]
                print(f"Found {len(raw_fields)} RAW-specific fields")
                
                # Print some examples
                if raw_fields:
                    print(f"\nExample RAW-specific fields:")
                    for field in sorted(raw_fields)[:5]:  # Show first 5 fields
                        value = raw_data[field]
                        if isinstance(value, (list, dict)) and len(str(value)) > 100:
                            value = str(value)[:100] + "..."
                        print(f"  {field}: {value}")
                
                # Test makernote tags
                print("\n--- Testing MakerNote tags ---")
                makernote_tags = specific_extractor.get_makernote_tags()
                print(f"Found {len(makernote_tags)} MakerNote tags")
                
                # Print some examples
                if makernote_tags:
                    print(f"\nExample MakerNote tags:")
                    count = 0
                    for tag, field in list(makernote_tags.items())[:5]:  # Show first 5 tags
                        print(f"  {tag} -> {field}")
                        count += 1
                    
                    if len(makernote_tags) > 5:
                        print(f"  ... and {len(makernote_tags) - 5} more tags")
                
                print(f"\nTEST PASSED: {extractor_class.__name__} for {os.path.basename(image_path)}")
            else:
                print(f"TEST FAILED: No {prefix.strip('_')}-specific fields found")
        except Exception as e:
            print(f"Error testing {extractor_class.__name__}: {str(e)}")
    else:
        print(f"No specific extractor found for {camera_make}")
    
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
