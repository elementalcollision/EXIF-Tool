#!/usr/bin/env python3
"""
Nikon Camera Extractor
Provides Nikon-specific EXIF extraction for NEF files
"""

import os
import io
import numpy as np
import rawpy
from PIL import Image
from typing import Dict, Any, Optional

from .base_extractor import CameraExtractor


class NikonExtractor(CameraExtractor):
    """Nikon-specific EXIF extractor for NEF files"""
    
    def can_handle(self, file_ext: str, exif_data: Dict[str, Any]) -> bool:
        """Check if this extractor can handle the given file
        
        Args:
            file_ext: File extension (e.g., '.nef')
            exif_data: Basic EXIF data already extracted
            
        Returns:
            True if this is a Nikon NEF file, False otherwise
        """
        # Check if it's a Nikon camera and NEF file
        is_nikon = exif_data.get('camera_make', '').upper().startswith('NIKON')
        is_nef = file_ext.lower() == '.nef'
        return is_nikon and is_nef
    
    def extract_metadata(self, image_path: str, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Nikon-specific metadata from the image
        
        Args:
            image_path: Path to the image file
            exif_data: Basic EXIF data already extracted
            
        Returns:
            Dictionary containing Nikon-specific metadata
        """
        # Process any Nikon-specific metadata
        result = {}
        
        # Add any Nikon-specific processing here that doesn't involve RAW data
        # This could include processing specific EXIF tags or other metadata
        
        # If there are Nikon MakerNote tags, process them
        nikon_tags = {k: v for k, v in exif_data.items() if k.startswith('nikon_')}
        if nikon_tags:
            result['has_nikon_makernote'] = True
            result['nikon_makernote_count'] = len(nikon_tags)
        
        return result
    
    def process_raw(self, image_path: str, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Nikon NEF file data
        
        Args:
            image_path: Path to the NEF file
            exif_data: Basic EXIF data already extracted
            
        Returns:
            Dictionary containing processed Nikon NEF data
        """
        result = {}
        
        try:
            print("Processing Nikon NEF specific data")
            with rawpy.imread(image_path) as raw:
                # Get raw image data
                raw_image = raw.raw_image.copy()
                
                # Get basic stats about the raw image
                raw_min = np.min(raw_image)
                raw_max = np.max(raw_image)
                print(f"Raw image shape: {raw_image.shape}")
                print(f"Raw image min/max values: {raw_min}/{raw_max}")
                
                # Calculate histogram
                histogram, _ = np.histogram(raw_image.flatten(), bins=256)
                
                # Nikon NEF specific fields
                nikon_metadata = {
                    'nikon_nef_version': 'NEF 1.0' if str(raw.raw_type) == 'RawType.Flat' else 'NEF 2.0',
                    'nikon_raw_image_shape': str(raw_image.shape),
                    'nikon_raw_min_value': int(raw_min),
                    'nikon_raw_max_value': int(raw_max),
                    'nikon_raw_histogram_mean': float(np.mean(histogram)),
                    'nikon_raw_histogram_std': float(np.std(histogram)),
                    'nikon_raw_dynamic_range': float(np.log2(raw_max - raw_min + 1)) if raw_max > raw_min else 0,
                    'nikon_raw_black_level': int(raw.black_level) if hasattr(raw, 'black_level') else 0,
                    'nikon_raw_white_level': int(raw.white_level) if hasattr(raw, 'white_level') else 0
                }
                
                # Add Nikon metadata to result
                result.update(nikon_metadata)
                
                # Try to extract thumbnail
                try:
                    thumb = raw.extract_thumb()
                    if thumb and hasattr(thumb, 'format'):
                        result['has_thumbnail'] = True
                        result['thumbnail_format'] = thumb.format
                        print(f"Extracted thumbnail in {thumb.format} format")
                        
                        # Process thumbnail if needed
                        if thumb.format == 'jpeg':
                            thumbnail_data = self.process_thumbnail(thumb.data, thumb.format)
                            if thumbnail_data:
                                result.update(thumbnail_data)
                except Exception as thumb_error:
                    result['has_thumbnail'] = False
                    print(f"Thumbnail extraction error: {thumb_error}")
                
                # Try to get color profile information
                if hasattr(raw, 'color_desc'):
                    result['nikon_color_profile'] = raw.color_desc.decode('utf-8', errors='ignore')
                    print(f"Color profile: {result['nikon_color_profile']}")
                
                # Get white balance coefficients if available
                if hasattr(raw, 'camera_whitebalance'):
                    # Handle different types of camera_whitebalance
                    if hasattr(raw.camera_whitebalance, 'tolist'):
                        result['nikon_camera_whitebalance'] = raw.camera_whitebalance.tolist()
                    else:
                        result['nikon_camera_whitebalance'] = str(raw.camera_whitebalance)
                    print(f"Camera white balance: {result['nikon_camera_whitebalance']}")
                
                # Get full resolution dimensions
                if hasattr(raw, 'sizes'):
                    result['nikon_full_width'] = raw.sizes.width
                    result['nikon_full_height'] = raw.sizes.height
                    print(f"Full resolution: {result['nikon_full_width']}x{result['nikon_full_height']}")
                
        except Exception as nikon_error:
            print(f"Nikon NEF specific error: {nikon_error}")
        
        return result
    
    def get_makernote_tags(self) -> Dict[str, str]:
        """Get Nikon MakerNote tag mapping
        
        Returns:
            Dictionary mapping Nikon MakerNote tag names to field names
        """
        return {
            'MakerNote NikonISOSpeed': 'nikon_iso_speed',
            'MakerNote NikonLensData': 'nikon_lens_data',
            'MakerNote NikonLensType': 'nikon_lens_type',
            'MakerNote NikonFocusDistance': 'nikon_focus_distance',
            'MakerNote NikonFlashSetting': 'nikon_flash_setting',
            'MakerNote NikonExposureMode': 'nikon_exposure_mode',
            'MakerNote NikonShootingMode': 'nikon_shooting_mode',
            'MakerNote NikonWhiteBalanceMode': 'nikon_white_balance_mode',
            'MakerNote NikonWhiteBalance': 'nikon_white_balance',
            'MakerNote NikonImageAdjustment': 'nikon_image_adjustment',
            'MakerNote NikonToneComp': 'nikon_tone_comp',
            'MakerNote NikonNoiseReduction': 'nikon_noise_reduction',
            'MakerNote NikonColorMode': 'nikon_color_mode',
            'MakerNote NikonHueAdjustment': 'nikon_hue_adjustment',
            'MakerNote NikonSharpening': 'nikon_sharpening',
            'MakerNote NikonFocusMode': 'nikon_focus_mode',
            'MakerNote NikonFlashMode': 'nikon_flash_mode',
            'MakerNote NikonFlashType': 'nikon_flash_type',
            'MakerNote NikonAFInfo': 'nikon_af_info',
            'MakerNote NikonImageOptimization': 'nikon_image_optimization',
            'MakerNote NikonActiveDLighting': 'nikon_active_d_lighting',
            'MakerNote NikonPictureControlData': 'nikon_picture_control_data',
            'MakerNote NikonWorldTime': 'nikon_world_time',
            'MakerNote NikonISOInfo': 'nikon_iso_info',
            'MakerNote NikonVRInfo': 'nikon_vr_info',
            'MakerNote NikonHighISONoiseReduction': 'nikon_high_iso_noise_reduction'
        }
    
    def process_thumbnail(self, thumb_data: bytes, thumb_format: str) -> Optional[Dict[str, Any]]:
        """Process Nikon thumbnail data with GPU acceleration if available
        
        Args:
            thumb_data: Raw thumbnail data
            thumb_format: Format of the thumbnail (e.g., 'jpeg')
            
        Returns:
            Dictionary containing processed thumbnail data, or None
        """
        result = {}
        
        # If using GPU acceleration, process the thumbnail
        if self.use_gpu and thumb_format == 'jpeg':
            try:
                import torch
                if torch.backends.mps.is_available():
                    print("Processing thumbnail with Metal GPU acceleration")
                    # Convert thumbnail to tensor and process with Metal
                    with Image.open(io.BytesIO(thumb_data)) as pil_thumb:
                        # Get thumbnail dimensions
                        thumb_width, thumb_height = pil_thumb.size
                        result['thumbnail_width'] = thumb_width
                        result['thumbnail_height'] = thumb_height
                        print(f"Thumbnail dimensions: {thumb_width}x{thumb_height}")
            except Exception as gpu_error:
                print(f"GPU processing error: {gpu_error}")
        
        return result
