#!/usr/bin/env python3
"""
Test script to debug EXIF extraction issues
"""

import os
import sys
import exifread
from PIL import Image
import rawpy
import datetime
import io

def test_exif_extraction(image_path):
    """Test EXIF extraction from a single image file"""
    print(f"\nTesting EXIF extraction for: {image_path}")
    
    # Check if file exists
    if not os.path.isfile(image_path):
        print(f"File does not exist: {image_path}")
        return
    
    # Get file info
    file_size = os.path.getsize(image_path)
    file_ext = os.path.splitext(image_path)[1].lower()
    print(f"File size: {file_size / 1024:.1f} KB")
    print(f"File extension: {file_ext}")
    
    # Method 1: Try PIL
    print("\n--- PIL Method ---")
    try:
        with Image.open(image_path) as img:
            print(f"PIL identified format: {img.format}")
            print(f"Image dimensions: {img.size}")
            
            if hasattr(img, '_getexif') and img._getexif():
                exif = img._getexif()
                print(f"PIL found {len(exif)} EXIF tags")
                
                # Print some common EXIF tags
                exif_tags = {
                    271: 'Make',
                    272: 'Model',
                    306: 'DateTime',
                    36867: 'DateTimeOriginal',
                    33434: 'ExposureTime',
                    33437: 'FNumber'
                }
                
                for tag_id, tag_name in exif_tags.items():
                    if tag_id in exif:
                        print(f"  {tag_name}: {exif[tag_id]}")
            else:
                print("No EXIF data found with PIL")
    except Exception as e:
        print(f"PIL error: {e}")
    
    # Method 2: Try exifread
    print("\n--- ExifRead Method ---")
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            if tags:
                print(f"exifread found {len(tags)} tags")
                
                # Print some common tags
                common_tags = [
                    'Image Make',
                    'Image Model',
                    'EXIF DateTimeOriginal',
                    'EXIF ExposureTime',
                    'EXIF FNumber'
                ]
                
                for tag in common_tags:
                    if tag in tags:
                        print(f"  {tag}: {tags[tag]}")
            else:
                print("No tags found with exifread")
    except Exception as e:
        print(f"exifread error: {e}")
    
    # Method 3: Try RAW processing for RAW files
    if file_ext in ['.arw', '.raw', '.nef', '.cr2', '.orf', '.rw2']:
        print("\n--- RAW Processing Method ---")
        try:
            with rawpy.imread(image_path) as raw:
                print(f"RAW type: {raw.raw_type}")
                if hasattr(raw, 'sizes'):
                    print(f"RAW dimensions: {raw.sizes.width}x{raw.sizes.height}")
                if hasattr(raw, 'color_desc'):
                    print(f"Color description: {raw.color_desc}")
                
                # For Sony ARW files
                if file_ext == '.arw':
                    print("\n--- Sony ARW Specific Info ---")
                    try:
                        raw_image = raw.raw_image
                        print(f"Raw image shape: {raw_image.shape}")
                        print(f"Raw image min/max values: {raw_image.min()}/{raw_image.max()}")
                        
                        if hasattr(raw, 'black_level'):
                            print(f"Black level: {raw.black_level}")
                        if hasattr(raw, 'white_level'):
                            print(f"White level: {raw.white_level}")
                    except Exception as sony_error:
                        print(f"Sony ARW specific error: {sony_error}")
        except Exception as raw_error:
            print(f"RAW processing error: {raw_error}")
    
    # Method 4: File system metadata
    print("\n--- File System Method ---")
    try:
        file_stat = os.stat(image_path)
        created = datetime.datetime.fromtimestamp(file_stat.st_ctime)
        modified = datetime.datetime.fromtimestamp(file_stat.st_mtime)
        print(f"File created: {created}")
        print(f"File modified: {modified}")
    except Exception as stat_error:
        print(f"File stat error: {stat_error}")

def main():
    """Main function"""
    # If a path is provided, use it
    if len(sys.argv) > 1:
        test_exif_extraction(sys.argv[1])
        return
    
    # Otherwise, find a sample image
    print("No image path provided. Looking for sample images...")
    sample_dirs = [
        '/Users/dave/Pictures',
        '.'
    ]
    
    for sample_dir in sample_dirs:
        if not os.path.isdir(sample_dir):
            continue
            
        for root, _, files in os.walk(sample_dir, topdown=True):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.arw', '.nef')):
                    image_path = os.path.join(root, file)
                    print(f"Found sample image: {image_path}")
                    test_exif_extraction(image_path)
                    return
    
    print("No sample images found.")

if __name__ == "__main__":
    main()
