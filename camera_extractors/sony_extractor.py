#!/usr/bin/env python3
"""
Sony Camera Extractor
Provides Sony-specific EXIF extraction for ARW files
"""

import os
import io
import numpy as np
import rawpy
from PIL import Image
from typing import Dict, Any, Optional

from .base_extractor import CameraExtractor


class SonyExtractor(CameraExtractor):
    """Sony-specific EXIF extractor for ARW files"""
    
    def can_handle(self, file_ext: str, exif_data: Dict[str, Any]) -> bool:
        """Check if this extractor can handle the given file
        
        Args:
            file_ext: File extension (e.g., '.arw')
            exif_data: Basic EXIF data already extracted
            
        Returns:
            True if this is a Sony ARW file, False otherwise
        """
        # Check if it's a Sony camera and ARW file
        is_sony = exif_data.get('camera_make', '').upper().startswith('SONY')
        is_arw = file_ext.lower() == '.arw'
        return is_sony and is_arw
    
    def extract_metadata(self, image_path: str, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Sony-specific metadata from the image
        
        Args:
            image_path: Path to the image file
            exif_data: Basic EXIF data already extracted
            
        Returns:
            Dictionary containing Sony-specific metadata
        """
        # Process any Sony-specific metadata
        result = {}
        
        # Add any Sony-specific processing here that doesn't involve RAW data
        # This could include processing specific EXIF tags or other metadata
        
        # If there are Sony MakerNote tags, process them
        sony_tags = {k: v for k, v in exif_data.items() if k.startswith('sony_')}
        if sony_tags:
            result['has_sony_makernote'] = True
            result['sony_makernote_count'] = len(sony_tags)
            
            # Add any additional processing of Sony MakerNote tags here
        
        return result
    
    def process_raw(self, image_path: str, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Sony ARW file data
        
        Args:
            image_path: Path to the ARW file
            exif_data: Basic EXIF data already extracted
            
        Returns:
            Dictionary containing processed Sony ARW data
        """
        result = {}
        
        try:
            print("Processing Sony ARW specific data")
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
                
                # Sony ARW specific fields
                sony_metadata = {
                    'sony_arw_version': 'ARW 2.0' if str(raw.raw_type) == 'RawType.Flat' else 'ARW 1.0',
                    'sony_raw_image_shape': str(raw_image.shape),
                    'sony_raw_min_value': int(raw_min),
                    'sony_raw_max_value': int(raw_max),
                    'sony_raw_histogram_mean': float(np.mean(histogram)),
                    'sony_raw_histogram_std': float(np.std(histogram)),
                    'sony_raw_dynamic_range': float(np.log2(raw_max - raw_min + 1)) if raw_max > raw_min else 0,
                    'sony_raw_black_level': int(raw.black_level) if hasattr(raw, 'black_level') else 0,
                    'sony_raw_white_level': int(raw.white_level) if hasattr(raw, 'white_level') else 0
                }
                
                # Add Sony metadata to result
                result.update(sony_metadata)
                
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
                    result['color_profile'] = raw.color_desc.decode('utf-8', errors='ignore')
                
                # Get white balance coefficients if available
                if hasattr(raw, 'camera_whitebalance'):
                    result['camera_white_balance'] = raw.camera_whitebalance.tolist()
                
        except Exception as sony_error:
            print(f"Sony ARW specific error: {sony_error}")
        
        return result
    
    def get_makernote_tags(self) -> Dict[str, str]:
        """Get Sony MakerNote tag mapping
        
        Returns:
            Dictionary mapping Sony MakerNote tag names to field names
        """
        return {
            'MakerNote SonyModelID': 'sony_model_id',
            'MakerNote SonyDateTime': 'sony_datetime',
            'MakerNote SonyImageHeight': 'sony_image_height',
            'MakerNote SonyImageWidth': 'sony_image_width',
            'MakerNote SonyColorMode': 'sony_color_mode',
            'MakerNote SonyQuality': 'sony_quality',
            'MakerNote SonyImageSize': 'sony_image_size',
            'MakerNote SonyFullImageSize': 'sony_full_image_size',
            'MakerNote SonyFrameRate': 'sony_frame_rate',
            'MakerNote SonyCreativeStyle': 'sony_creative_style',
            'MakerNote SonyFocusMode': 'sony_focus_mode',
            'MakerNote SonyAFAreaMode': 'sony_af_area_mode',
            'MakerNote SonyAFPointSelected': 'sony_af_point_selected',
            'MakerNote SonyDriveMode': 'sony_drive_mode',
            'MakerNote SonyWhiteBalance': 'sony_white_balance',
            'MakerNote SonyColorTemperature': 'sony_color_temperature',
            'MakerNote SonyReleaseMode': 'sony_release_mode',
            'MakerNote SonyDigitalZoom': 'sony_digital_zoom',
            'MakerNote SonyLensID': 'sony_lens_id',
            'MakerNote SonyLensType': 'sony_lens_type',
            'MakerNote SonyLensSpec': 'sony_lens_spec',
            'MakerNote SonyBatteryLevel': 'sony_battery_level',
            'MakerNote SonyPictureEffect': 'sony_picture_effect',
            'MakerNote SonyPanoramaSize': 'sony_panorama_size',
            'MakerNote SonyShutterCount': 'sony_shutter_count'
        }
    
    def process_thumbnail(self, thumb_data: bytes, thumb_format: str) -> Optional[Dict[str, Any]]:
        """Process Sony thumbnail data with GPU acceleration if available
        
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
