#!/usr/bin/env python3
"""
Apple RAW Camera Extractor
Provides Apple ProRAW-specific EXIF extraction with support for iPhone models
"""

import os
import io
import numpy as np
import rawpy
import json
import subprocess
from PIL import Image
from typing import Dict, Any, Optional, List, Tuple
import platform

from .base_extractor import CameraExtractor
from .dng_extractor import DngExtractor

class AppleRawExtractor(DngExtractor):
    """Apple ProRAW-specific EXIF extractor with support for iPhone models"""
    
    def can_handle(self, file_ext: str, exif_data: Dict[str, Any]) -> bool:
        """Check if this extractor can handle the given file
        
        Args:
            file_ext: File extension (e.g., '.dng')
            exif_data: Basic EXIF data already extracted
            
        Returns:
            True if this is an Apple ProRAW file, False otherwise
        """
        # Check if it's a DNG file (Apple ProRAW uses DNG format)
        is_dng = file_ext.lower() == '.dng'
        
        # Check if it's from an Apple device (iPhone)
        is_apple = False
        if is_dng:
            make = exif_data.get('camera_make', '').upper()
            model = exif_data.get('camera_model', '').upper()
            
            # Check for Apple/iPhone identifiers
            if 'APPLE' in make or 'IPHONE' in model:
                print(f"Detected Apple ProRAW file from {model}")
                is_apple = True
        
        return is_dng and is_apple
    
    def extract_metadata(self, image_path: str, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Apple ProRAW-specific metadata from the image
        
        Args:
            image_path: Path to the image file
            exif_data: Basic EXIF data already extracted
            
        Returns:
            Dictionary containing Apple ProRAW-specific metadata
        """
        # First get the standard DNG metadata using the parent class
        result = super().extract_metadata(image_path, exif_data)
        
        # Try to extract Apple-specific metadata using exiftool if available
        try:
            if self._check_exiftool():
                print("Using exiftool to extract Apple ProRAW metadata")
                exiftool_data = self._run_exiftool(image_path)
                if exiftool_data:
                    # Process Apple-specific tags
                    for key, value in exiftool_data.items():
                        if key.startswith('Apple') or 'Apple' in key:
                            # Convert to snake_case
                            apple_key = 'apple_' + key.replace('Apple', '').lower().replace(' ', '_')
                            result[apple_key] = value
                    
                    # Extract computational photography data
                    if 'ComputationalPhotography' in exiftool_data:
                        comp_photo = exiftool_data.get('ComputationalPhotography', {})
                        if isinstance(comp_photo, dict):
                            for cp_key, cp_value in comp_photo.items():
                                result[f'apple_comp_{cp_key.lower()}'] = cp_value
                    
                    # Extract HDR information
                    if 'HDR' in exiftool_data:
                        result['apple_hdr'] = exiftool_data.get('HDR')
                    
                    # Extract Deep Fusion information
                    if 'DeepFusion' in exiftool_data:
                        result['apple_deep_fusion'] = exiftool_data.get('DeepFusion')
        except Exception as e:
            print(f"Error extracting Apple ProRAW metadata: {e}")
        
        return result
    
    def process_raw(self, image_path: str, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Apple ProRAW file data
        
        Args:
            image_path: Path to the Apple ProRAW file
            exif_data: Basic EXIF data already extracted
            
        Returns:
            Dictionary containing processed Apple ProRAW data
        """
        result = {}
        
        try:
            # First try using the CoreImage API if on macOS
            if platform.system() == 'Darwin':
                print("Using macOS CoreImage API for Apple ProRAW processing")
                ci_data = self._process_with_coreimage(image_path)
                if ci_data:
                    result.update(ci_data)
            
            # If CoreImage processing didn't yield results or we're not on macOS,
            # fall back to standard rawpy processing
            if not result:
                print("Falling back to standard RAW processing for Apple ProRAW")
                # Use the parent class's process_raw method
                dng_result = super().process_raw(image_path, exif_data)
                if dng_result:
                    result.update(dng_result)
                    
                    # Add Apple-specific processing
                    with rawpy.imread(image_path) as raw:
                        # Check for Apple-specific metadata in the maker notes
                        if hasattr(raw, 'metadata') and raw.metadata:
                            apple_metadata = {}
                            for key, value in raw.metadata.items():
                                if 'apple' in key.lower():
                                    apple_metadata[f'apple_{key.lower()}'] = value
                            result.update(apple_metadata)
                        
                        # Process the image with Apple-specific demosaicing settings
                        try:
                            # Use natural demosaicing for Apple ProRAW
                            rgb = raw.postprocess(demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
                                                use_camera_wb=True,
                                                no_auto_bright=True,
                                                output_color=rawpy.ColorSpace.sRGB)
                            
                            # Calculate image statistics
                            result['apple_raw_mean_r'] = float(np.mean(rgb[:,:,0]))
                            result['apple_raw_mean_g'] = float(np.mean(rgb[:,:,1]))
                            result['apple_raw_mean_b'] = float(np.mean(rgb[:,:,2]))
                            result['apple_raw_std_r'] = float(np.std(rgb[:,:,0]))
                            result['apple_raw_std_g'] = float(np.std(rgb[:,:,1]))
                            result['apple_raw_std_b'] = float(np.std(rgb[:,:,2]))
                            
                            # Calculate dynamic range per channel
                            result['apple_raw_dynamic_range_r'] = float(np.log2(np.max(rgb[:,:,0]) - np.min(rgb[:,:,0]) + 1))
                            result['apple_raw_dynamic_range_g'] = float(np.log2(np.max(rgb[:,:,1]) - np.min(rgb[:,:,1]) + 1))
                            result['apple_raw_dynamic_range_b'] = float(np.log2(np.max(rgb[:,:,2]) - np.min(rgb[:,:,2]) + 1))
                            
                            print(f"Processed Apple ProRAW image with dimensions: {rgb.shape}")
                        except Exception as proc_error:
                            print(f"Error processing Apple ProRAW image: {proc_error}")
        
        except Exception as apple_error:
            print(f"Apple ProRAW specific error: {apple_error}")
        
        return result
    
    def _process_with_coreimage(self, image_path: str) -> Dict[str, Any]:
        """Process Apple ProRAW using macOS CoreImage API
        
        Args:
            image_path: Path to the Apple ProRAW file
            
        Returns:
            Dictionary containing CoreImage processed data
        """
        result = {}
        
        # Only available on macOS
        if platform.system() != 'Darwin':
            return result
        
        try:
            # Use PyObjC to access CoreImage API
            try:
                from Foundation import NSURL
                from Quartz import CIImage, CIContext, CIFilter
                
                # Create URL from file path
                url = NSURL.fileURLWithPath_(image_path)
                
                # Create CIImage from URL
                ci_image = CIImage.imageWithContentsOfURL_(url)
                
                if ci_image:
                    # Get image properties
                    properties = ci_image.properties()
                    
                    # Extract Apple ProRAW specific properties
                    if properties:
                        for key, value in properties.items():
                            if 'apple' in key.lower() or 'proraw' in key.lower():
                                result[f'apple_ci_{key.lower()}'] = str(value)
                        
                        # Extract dimensions
                        extent = ci_image.extent()
                        result['apple_ci_width'] = extent.size.width
                        result['apple_ci_height'] = extent.size.height
                        
                        # Extract color space information
                        if 'ColorModel' in properties:
                            result['apple_ci_color_model'] = properties['ColorModel']
                        
                        # Extract ProRAW specific metadata
                        if '{TIFF}' in properties:
                            tiff_dict = properties['{TIFF}']
                            for tiff_key, tiff_value in tiff_dict.items():
                                if 'apple' in tiff_key.lower():
                                    result[f'apple_ci_tiff_{tiff_key.lower()}'] = str(tiff_value)
                    
                    print("Successfully processed Apple ProRAW with CoreImage")
            except ImportError:
                print("PyObjC not available, skipping CoreImage processing")
        
        except Exception as ci_error:
            print(f"CoreImage processing error: {ci_error}")
        
        return result
    
    def get_makernote_tags(self) -> Dict[str, str]:
        """Get Apple ProRAW MakerNote tag mapping
        
        Returns:
            Dictionary mapping Apple ProRAW MakerNote tag names to field names
        """
        # Get the base DNG makernote tags
        makernote_tags = super().get_makernote_tags()
        
        # Add Apple-specific tags
        apple_tags = {
            'Apple ComputationalPhotography': 'apple_computational_photography',
            'Apple ProRAW': 'apple_proraw',
            'Apple DeepFusion': 'apple_deep_fusion',
            'Apple NightMode': 'apple_night_mode',
            'Apple HDR': 'apple_hdr',
            'Apple SmartHDR': 'apple_smart_hdr',
            'Apple PhotonicsEngine': 'apple_photonics_engine',
            'Apple LocalToneMapping': 'apple_local_tone_mapping',
            'Apple FusionVersion': 'apple_fusion_version',
            'Apple ImageProcessor': 'apple_image_processor'
        }
        makernote_tags.update(apple_tags)
        
        return makernote_tags
