#!/usr/bin/env python3
"""
Test script for EXIF database with Sony ARW files
"""

import os
import sys
from exif_extractor import EnhancedExifExtractor
from exif_db import ExifDatabase

def test_with_db(image_path, use_gpu=False):
    """Test EXIF extraction with database storage
    
    Args:
        image_path: Path to the image file
        use_gpu: Whether to use GPU acceleration
    """
    print(f"\n=== Testing EXIF extraction with database for: {image_path} ===\n")
    
    # Create an in-memory database for testing
    db = ExifDatabase(":memory:")
    
    # Create extractor with resource management settings
    extractor = EnhancedExifExtractor(
        use_gpu=use_gpu,
        cpu_cores=None,  # Auto-detect (n-2 on Apple Silicon)
        memory_limit_percent=75  # Cap memory usage at 75% of available RAM
    )
    
    # Extract EXIF data
    exif_data = extractor.extract_exif(image_path)
    
    if not exif_data:
        print("Failed to extract EXIF data")
        return
    
    print(f"\nExtracted {len(exif_data)} EXIF fields")
    
    # Store in database
    image_id = db.add_image(exif_data)
    print(f"Added to database with ID: {image_id}")
    
    # Create a collection
    collection_name = os.path.basename(os.path.dirname(image_path))
    collection_id = db.create_collection(
        name=collection_name,
        description=f"Test collection for {collection_name}"
    )
    
    # Add image to collection
    db.add_image_to_collection(collection_id, image_id)
    print(f"Added to collection '{collection_name}' with ID: {collection_id}")
    
    # Retrieve from database
    retrieved_data = db.get_image(image_id)
    print(f"Retrieved {len(retrieved_data)} fields from database")
    
    # Print the most important fields
    print("\nKey EXIF data:")
    priority_fields = [
        'file_name', 'camera_make', 'camera_model', 'date_taken', 
        'exposure_time', 'f_number', 'iso', 'focal_length',
        'width', 'height', 'sony_arw_version', 'sony_raw_black_level', 'sony_raw_white_level'
    ]
    
    for field in priority_fields:
        if field in retrieved_data:
            print(f"{field}: {retrieved_data[field]}")
    
    # Get database stats
    stats = db.get_stats()
    print("\nDatabase Statistics:")
    print(f"Total images: {stats['total_images']}")
    print(f"Images by camera make: {stats['images_by_camera_make']}")
    print(f"Images by file type: {stats['images_by_file_type']}")
    
    # Close database
    db.close()
    print("\nTest completed successfully")


if __name__ == "__main__":
    # If a path is provided, use it
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        # Check if GPU flag is provided
        use_gpu = "--gpu" in sys.argv
        test_with_db(image_path, use_gpu)
    else:
        print("Usage: python test_db_exif.py <image_path> [--gpu]")
