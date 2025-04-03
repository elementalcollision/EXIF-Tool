#!/usr/bin/env python3
"""
Test script for Panasonic RAW extractor
Tests the extraction of Panasonic-specific metadata from RW2 files
"""

import os
import sys
import json
from typing import Dict, Any
from exif_extractor import EnhancedExifExtractor
from camera_extractors import PanasonicExtractor

def test_panasonic_raw(image_path: str) -> None:
    """
    Test Panasonic RAW extraction on a specific image
    
    Args:
        image_path: Path to the Panasonic RW2 image
    """
    print(f"Testing Panasonic RAW extraction on: {image_path}")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return
    
    # Check file extension
    _, ext = os.path.splitext(image_path)
    if ext.lower() != '.rw2':
        print(f"Warning: File does not have .RW2 extension: {ext}")
    
    # Create extractor
    extractor = EnhancedExifExtractor()
    
    # Extract EXIF data
    exif_data = extractor.extract_exif(image_path)
    
    # Check if it's a Panasonic camera
    camera_make = exif_data.get('camera_make', '')
    if camera_make.upper() != 'PANASONIC':
        print(f"Warning: Not a Panasonic camera: {camera_make}")
    
    # Print basic EXIF data
    print("\nBasic EXIF Data:")
    print(f"Camera Make: {exif_data.get('camera_make', 'Unknown')}")
    print(f"Camera Model: {exif_data.get('camera_model', 'Unknown')}")
    print(f"Lens: {exif_data.get('lens', 'Unknown')}")
    print(f"Focal Length: {exif_data.get('focal_length', 'Unknown')}")
    print(f"Aperture: {exif_data.get('aperture', 'Unknown')}")
    print(f"ISO: {exif_data.get('iso', 'Unknown')}")
    print(f"Shutter Speed: {exif_data.get('shutter_speed', 'Unknown')}")
    
    # Print Panasonic-specific data
    print("\nPanasonic-specific Data:")
    panasonic_fields = [key for key in exif_data.keys() if key.startswith('panasonic_')]
    
    if not panasonic_fields:
        print("No Panasonic-specific data found!")
    else:
        for field in sorted(panasonic_fields):
            print(f"{field}: {exif_data[field]}")
    
    # Test direct Panasonic extractor
    print("\nTesting direct Panasonic extractor:")
    panasonic_extractor = PanasonicExtractor()
    
    # Check if extractor can handle this file
    can_handle = panasonic_extractor.can_handle(ext, exif_data)
    print(f"Can handle this file: {can_handle}")
    
    if can_handle:
        # Extract metadata directly
        panasonic_data = panasonic_extractor.extract_metadata(image_path, exif_data)
        
        # Print additional fields found by direct extraction
        print("\nAdditional fields from direct extraction:")
        additional_fields = set(panasonic_data.keys()) - set(exif_data.keys())
        
        if not additional_fields:
            print("No additional fields found")
        else:
            for field in sorted(additional_fields):
                print(f"{field}: {panasonic_data[field]}")
    
    print("\nExtraction test completed")

def main() -> None:
    """Main function"""
    # Check command line arguments
    if len(sys.argv) < 2:
        # Use default test file if available
        test_file = os.path.join("Test_for_EXIF", "RAW_PANASONIC_DMC-GH4.RW2")
        if not os.path.exists(test_file):
            print("Usage: python test_panasonic_raw.py <path_to_rw2_file>")
            return
    else:
        test_file = sys.argv[1]
    
    # Run test
    test_panasonic_raw(test_file)

if __name__ == "__main__":
    main()
