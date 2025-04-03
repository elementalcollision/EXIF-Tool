#!/usr/bin/env python3
"""
Template Camera Extractor
Template for creating new camera-specific extractors
"""

import os
from typing import Dict, Any, Optional

from .base_extractor import CameraExtractor


class TemplateExtractor(CameraExtractor):
    """Template for creating new camera-specific extractors
    
    To create a new camera extractor:
    1. Copy this file and rename it (e.g., nikon_extractor.py)
    2. Rename the class (e.g., NikonExtractor)
    3. Implement the required methods
    4. Register the extractor in __init__.py
    """
    
    def can_handle(self, file_ext: str, exif_data: Dict[str, Any]) -> bool:
        """Check if this extractor can handle the given file
        
        Args:
            file_ext: File extension (e.g., '.nef')
            exif_data: Basic EXIF data already extracted
            
        Returns:
            True if this extractor can handle the file, False otherwise
        """
        # Example implementation for Nikon NEF files:
        # is_nikon = exif_data.get('camera_make', '').upper().startswith('NIKON')
        # is_nef = file_ext.lower() == '.nef'
        # return is_nikon and is_nef
        return False
    
    def extract_metadata(self, image_path: str, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract camera-specific metadata from the image
        
        Args:
            image_path: Path to the image file
            exif_data: Basic EXIF data already extracted
            
        Returns:
            Dictionary containing camera-specific metadata
        """
        # Process any camera-specific metadata
        result = {}
        
        # Add any camera-specific processing here that doesn't involve RAW data
        # This could include processing specific EXIF tags or other metadata
        
        return result
    
    def process_raw(self, image_path: str, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process RAW file data for this camera type
        
        Args:
            image_path: Path to the image file
            exif_data: Basic EXIF data already extracted
            
        Returns:
            Dictionary containing the processed RAW data
        """
        result = {}
        
        try:
            # Add camera-specific RAW processing here
            # Example:
            # with rawpy.imread(image_path) as raw:
            #     # Process RAW data
            #     ...
            pass
        except Exception as error:
            print(f"RAW processing error: {error}")
        
        return result
    
    def get_makernote_tags(self) -> Dict[str, str]:
        """Get MakerNote tag mapping for this camera type
        
        Returns:
            Dictionary mapping MakerNote tag names to field names
        """
        # Example for Nikon:
        # return {
        #     'MakerNote NikonModel': 'nikon_model',
        #     'MakerNote NikonLensData': 'nikon_lens_data',
        #     # Add more MakerNote tags as needed
        # }
        return {}
    
    def process_thumbnail(self, thumb_data: bytes, thumb_format: str) -> Optional[Dict[str, Any]]:
        """Process thumbnail data (optional, can be overridden by subclasses)
        
        Args:
            thumb_data: Raw thumbnail data
            thumb_format: Format of the thumbnail (e.g., 'jpeg')
            
        Returns:
            Dictionary containing processed thumbnail data, or None
        """
        # Override this method if you need custom thumbnail processing
        # By default, use the base class implementation
        return super().process_thumbnail(thumb_data, thumb_format)
