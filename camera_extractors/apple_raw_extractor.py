#!/usr/bin/env python3
"""
Apple RAW Camera Extractor
Provides Apple ProRAW-specific EXIF extraction with support for iPhone models
"""

import os
import io
import re
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
    
    def __init__(self, use_gpu=False, memory_limit=None, cpu_cores=None):
        """Initialize the Apple RAW extractor
        
        Args:
            use_gpu: Whether to use GPU acceleration
            memory_limit: Memory limit in bytes
            cpu_cores: Number of CPU cores to use
        """
        super().__init__(use_gpu=use_gpu)
        self.is_leica = False  # Initialize is_leica attribute
        self.memory_limit = memory_limit
        self.cpu_cores = cpu_cores
    
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
                    
                    # Extract computational photography features
                    comp_photo_features = {
                        'HDR': 'apple_hdr',
                        'DeepFusion': 'apple_deep_fusion',
                        'NightMode': 'apple_night_mode',
                        'SmartHDR': 'apple_smart_hdr',
                        'PhotonicsEngine': 'apple_photonics_engine',
                        'LocalToneMapping': 'apple_local_tone_mapping',
                        'ProRAW': 'apple_proraw_enabled',
                        'ComputationalPhotography': 'apple_comp_photo_enabled'
                    }
                    
                    for feature, field_name in comp_photo_features.items():
                        if feature in exiftool_data:
                            result[field_name] = exiftool_data.get(feature)
                    
                    # Extract iPhone model-specific features
                    if 'iPhone' in exif_data.get('camera_model', ''):
                        model = exif_data.get('camera_model', '')
                        # Extract model number
                        if 'iPhone 13' in model:
                            result['apple_photographic_styles'] = True
                            result['apple_cinematic_mode'] = True
                        if 'iPhone 14' in model or 'iPhone 15' in model:
                            result['apple_photographic_styles'] = True
                            result['apple_cinematic_mode'] = True
                            result['apple_action_mode'] = True
                        if 'Pro' in model:
                            result['apple_macro_mode'] = True
                    
                    # Parse Apple maker notes for additional computational features
                    if '{MakerApple}' in exiftool_data:
                        maker_data = exiftool_data.get('{MakerApple}')
                        if isinstance(maker_data, str):
                            # Check for known computational photography indicators
                            if '33 =' in maker_data and 'flags' in maker_data:
                                result['apple_computational_pipeline'] = True
                            
                            # Extract fusion version if available
                            fusion_match = re.search(r'\d+\.\d+\.\d+', maker_data)
                            if fusion_match:
                                result['apple_fusion_version'] = fusion_match.group(0)
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
                            
                            # Analyze image for computational photography indicators
                            # Check for noise patterns characteristic of Deep Fusion
                            noise_levels = [np.std(rgb[i:i+100, j:j+100]) 
                                           for i in range(0, rgb.shape[0], 100) 
                                           for j in range(0, rgb.shape[1], 100) 
                                           if i+100 < rgb.shape[0] and j+100 < rgb.shape[1]]
                            
                            if noise_levels:
                                noise_variance = np.var(noise_levels)
                                result['apple_noise_variance'] = float(noise_variance)
                                
                                # Deep Fusion typically has very consistent noise patterns
                                if noise_variance < 0.01:
                                    result['apple_deep_fusion_detected'] = True
                                
                                # HDR typically has expanded dynamic range
                                if result.get('apple_raw_dynamic_range_r', 0) > 14 or \
                                   result.get('apple_raw_dynamic_range_g', 0) > 14 or \
                                   result.get('apple_raw_dynamic_range_b', 0) > 14:
                                    result['apple_hdr_detected'] = True
                            
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
