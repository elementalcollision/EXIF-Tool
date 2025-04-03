#!/usr/bin/env python3
"""
Test script for Olympus RAW Extractor
Tests the OlympusExtractor with sample ORF files
"""

import os
import sys
import json
import logging
import time
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_olympus')

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera_extractors import OlympusExtractor
from exif_extractor import EnhancedExifExtractor

def test_olympus_extractor(image_path: str, use_gpu: bool = False, memory_limit: float = 0.75) -> Dict[str, Any]:
    """Test the Olympus extractor with a given image
    
    Args:
        image_path: Path to the ORF file
        use_gpu: Whether to use GPU acceleration
        memory_limit: Memory limit as a fraction of available RAM
        
    Returns:
        Combined metadata from EXIF and Olympus-specific extraction
    """
    start_time = time.time()
    logger.info(f"Testing Olympus extractor on {image_path}")
    
    # First extract basic EXIF data
    exif_extractor = EnhancedExifExtractor()
    basic_exif = exif_extractor.extract_exif(image_path)
    
    # Now use the Olympus-specific extractor
    olympus_extractor = OlympusExtractor(use_gpu=use_gpu, memory_limit=memory_limit)
    
    # Check if the extractor can handle this file
    file_ext = os.path.splitext(image_path)[1].lower()
    can_handle = olympus_extractor.can_handle(file_ext, basic_exif)
    logger.info(f"Can Olympus extractor handle this file? {can_handle}")
    
    if not can_handle:
        logger.error("This file is not recognized as an Olympus ORF file")
        return basic_exif
    
    # Extract Olympus-specific metadata
    olympus_metadata = olympus_extractor.extract_metadata(image_path, basic_exif)
    logger.info(f"Extracted {len(olympus_metadata)} Olympus-specific metadata fields")
    
    # Process RAW data
    raw_data = olympus_extractor.process_raw(image_path, basic_exif)
    logger.info(f"Processed RAW data with {len(raw_data)} fields")
    
    # Combine all metadata
    combined_metadata = {**basic_exif, **olympus_metadata, **raw_data}
    
    # Log processing time
    end_time = time.time()
    logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
    
    return combined_metadata

def main():
    """Main function to run the test"""
    # Check if path is provided as argument
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        # Default test file
        test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Test_for_EXIF")
        test_file = os.path.join(test_dir, "P1161088.ORF")
    
    # Check if file exists
    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}")
        sys.exit(1)
    
    # Get command line arguments for GPU and memory limit
    use_gpu = "--gpu" in sys.argv
    memory_limit = 0.75  # Default
    
    for arg in sys.argv:
        if arg.startswith("--memory="):
            try:
                memory_limit = float(arg.split("=")[1])
                if memory_limit <= 0 or memory_limit > 1:
                    logger.warning(f"Invalid memory limit: {memory_limit}, using default 0.75")
                    memory_limit = 0.75
            except:
                logger.warning(f"Invalid memory limit format: {arg}, using default 0.75")
    
    logger.info(f"Testing with GPU: {use_gpu}, Memory limit: {memory_limit}")
    
    # Run the test
    result = test_olympus_extractor(test_file, use_gpu, memory_limit)
    
    # Print summary of results
    print("\n" + "="*50)
    print(f"OLYMPUS RAW EXTRACTOR TEST RESULTS")
    print("="*50)
    
    # Print basic camera info
    print(f"\nCamera: {result.get('camera_make', 'Unknown')} {result.get('camera_model', 'Unknown')}")
    print(f"File: {os.path.basename(test_file)}")
    
    # Print Olympus-specific fields
    olympus_fields = {k: v for k, v in result.items() if k.startswith('olympus_')}
    print(f"\nExtracted {len(olympus_fields)} Olympus-specific fields:")
    
    # Group fields by category
    categories = {
        "Camera Info": ["camera_series", "camera_type", "equipment"],
        "Image Settings": ["image_width", "image_height", "color_profile", "raw_"],
        "Focus": ["focus_"],
        "Processing": ["image_processing", "raw_development"],
        "Performance": ["field_count", "highlight_percentage", "shadow_percentage", "midtone_percentage"]
    }
    
    for category, keywords in categories.items():
        category_fields = {}
        for k, v in olympus_fields.items():
            if any(keyword in k for keyword in keywords):
                category_fields[k] = v
        
        if category_fields:
            print(f"\n{category}:")
            for k, v in sorted(category_fields.items()):
                # Format the output for better readability
                if isinstance(v, (list, dict)) or len(str(v)) > 50:
                    print(f"  {k}: [complex data]")
                else:
                    print(f"  {k}: {v}")
    
    # Save full results to JSON
    output_file = os.path.splitext(test_file)[0] + "_olympus_metadata.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\nFull metadata saved to: {output_file}")

if __name__ == "__main__":
    main()
