#!/usr/bin/env python3
"""
Test script to verify all camera extractors are functioning properly
Tests initialization and compatibility with different file formats
"""

import os
import sys
import unittest
from typing import Dict, Any, List

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from camera_extractors import get_camera_extractor
from camera_extractors.extractor_factory import _CAMERA_EXTRACTORS


class TestExtractorCompatibility(unittest.TestCase):
    """Test case for verifying camera extractor compatibility"""
    
    def setUp(self):
        """Set up the test case"""
        # Get list of available extractors from the registry
        self.available_extractors = list(_CAMERA_EXTRACTORS.keys())
        self.test_extensions = ['.nef', '.cr2', '.cr3', '.arw', '.dng', '.raf', '.rw2', '.orf', '.proraw']
        # Create a dummy exif_data dict for testing
        self.dummy_exif = {'camera_make': 'TEST', 'camera_model': 'TEST MODEL'}
    
    def test_extractor_initialization(self):
        """Test that all extractors can be initialized"""
        print(f"\nTesting initialization of {len(self.available_extractors)} camera extractors:")
        
        for camera_type in self.available_extractors:
            # Get the extractor class for this camera type
            extractor_class = _CAMERA_EXTRACTORS[camera_type]
            with self.subTest(camera_type=camera_type):
                try:
                    extractor_instance = extractor_class(use_gpu=False)
                    print(f"  ✓ Successfully initialized {extractor_class.__name__} for {camera_type}")
                    self.assertIsNotNone(extractor_instance)
                except Exception as e:
                    print(f"  ✗ Error initializing {extractor_class.__name__} for {camera_type}: {e}")
                    self.fail(f"Failed to initialize {extractor_class.__name__}: {e}")
    
    def test_file_extension_handling(self):
        """Test that file extensions are handled correctly"""
        print(f"\nTesting file extension handling for {len(self.test_extensions)} extensions:")
        
        for ext in self.test_extensions:
            # Find extractors that can handle this extension
            handlers = []
            with self.subTest(extension=ext):
                for camera_type in self.available_extractors:
                    extractor_class = _CAMERA_EXTRACTORS[camera_type]
                    try:
                        extractor_instance = extractor_class(use_gpu=False)
                        if extractor_instance.can_handle(ext, self.dummy_exif):
                            handlers.append(extractor_class.__name__)
                    except Exception as e:
                        print(f"  ✗ Error with {extractor_class.__name__} for {ext}: {e}")
                
                print(f"  - {ext}: {', '.join(handlers) if handlers else 'No handlers'}")
                # We don't assert here since some extensions might not have handlers
                # But we do want to see which ones do
    
    def test_specific_camera_handling(self):
        """Test that specific camera makes are handled correctly"""
        print("\nTesting camera make handling:")
        
        test_cameras = [
            {'camera_make': 'SONY', 'camera_model': 'A7R IV', 'file_ext': '.arw'},
            {'camera_make': 'NIKON', 'camera_model': 'Z9', 'file_ext': '.nef'},
            {'camera_make': 'CANON', 'camera_model': 'EOS R5', 'file_ext': '.cr3'},
            {'camera_make': 'LEICA', 'camera_model': 'M11', 'file_ext': '.dng'},
            {'camera_make': 'FUJIFILM', 'camera_model': 'X-T5', 'file_ext': '.raf'},
            {'camera_make': 'OLYMPUS', 'camera_model': 'OM-1', 'file_ext': '.orf'},
            {'camera_make': 'APPLE', 'camera_model': 'iPhone 15 Pro', 'file_ext': '.dng'},
            {'camera_make': 'PANASONIC', 'camera_model': 'LUMIX GH6', 'file_ext': '.rw2'},
        ]
        
        for camera in test_cameras:
            with self.subTest(camera=f"{camera['camera_make']} {camera['camera_model']}"):
                # Find extractors that can handle this camera
                handlers = []
                for camera_type in self.available_extractors:
                    extractor_class = _CAMERA_EXTRACTORS[camera_type]
                    try:
                        extractor_instance = extractor_class(use_gpu=False)
                        if extractor_instance.can_handle(camera['file_ext'], camera):
                            handlers.append(extractor_class.__name__)
                    except Exception as e:
                        print(f"  ✗ Error with {extractor_class.__name__} for {camera['camera_make']}: {e}")
                
                print(f"  - {camera['camera_make']} {camera['camera_model']} ({camera['file_ext']}): "
                      f"{', '.join(handlers) if handlers else 'No handlers'}")
                
                # Assert that each camera has at least one handler
                self.assertTrue(len(handlers) > 0, 
                               f"No handlers found for {camera['camera_make']} {camera['camera_model']}")


def run_tests():
    """Run the tests"""
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


if __name__ == "__main__":
    print("Testing camera extractor compatibility...")
    run_tests()
