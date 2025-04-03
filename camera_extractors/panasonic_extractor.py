#!/usr/bin/env python3
"""
Panasonic Camera RAW Extractor
Extracts metadata from Panasonic RW2 files
"""

import os
import json
import subprocess
import rawpy
import numpy as np
import time
import logging
import concurrent.futures
import gc
from typing import Dict, Any, List, Tuple, Optional, Callable
from .base_extractor import CameraExtractor, ThumbFormat
from camera_extractors.optimization_utils import (
    MemoryTracker, performance_monitor, safe_array_operation
)

# Configure logging
logger = logging.getLogger('panasonic_extractor')

class PanasonicExtractor(CameraExtractor):
    """
    Extractor for Panasonic RAW (RW2) files
    Supports Panasonic Lumix cameras
    """
    
    def __init__(self, use_gpu=False, memory_limit=0.75, cpu_cores=None):
        super().__init__(use_gpu=use_gpu, memory_limit=memory_limit, cpu_cores=cpu_cores)
        
        # Log initialization
        logger.info(f"Initializing PanasonicExtractor with GPU={use_gpu}, "
                  f"memory_limit={memory_limit}, cpu_cores={cpu_cores}")
        
        self.panasonic_models = {
            'DMC-GH4': {
                'sensor': 'MFT',
                'has_ibis': True,
                'supports_4k': True,
                'supports_vlog': True,
            },
            'DMC-GH5': {
                'sensor': 'MFT',
                'has_ibis': True,
                'supports_4k': True,
                'supports_vlog': True,
                'supports_6k_photo': True,
            },
            'DC-GH5S': {
                'sensor': 'MFT',
                'has_ibis': False,  # GH5S doesn't have IBIS
                'supports_4k': True,
                'supports_vlog': True,
                'dual_native_iso': True,
            },
            'DC-GH6': {
                'sensor': 'MFT',
                'has_ibis': True,
                'supports_4k': True,
                'supports_vlog': True,
                'supports_prores': True,
                'supports_cfexpress': True,
            },
            'DC-S1': {
                'sensor': 'Full-frame',
                'has_ibis': True,
                'supports_4k': True,
                'supports_vlog': True,
            },
            'DC-S1R': {
                'sensor': 'Full-frame',
                'has_ibis': True,
                'supports_4k': True,
                'high_resolution_mode': True,
            },
            'DC-S5': {
                'sensor': 'Full-frame',
                'has_ibis': True,
                'supports_4k': True,
                'supports_vlog': True,
            },
            'DMC-G9': {
                'sensor': 'MFT',
                'has_ibis': True,
                'supports_4k': True,
                'high_resolution_mode': True,
            },
            'DMC-LX100': {
                'sensor': 'MFT',
                'has_ibis': False,
                'supports_4k': True,
                'fixed_lens': True,
            },
        }
        
    @staticmethod
    def can_handle(file_ext: str, exif_data: Dict[str, Any]) -> bool:
        """
        Check if this extractor can handle the given file
        
        Args:
            file_ext: File extension
            exif_data: Basic EXIF data
            
        Returns:
            True if this extractor can handle the file, False otherwise
        """
        # Check if it's a Panasonic camera
        is_panasonic = exif_data.get('camera_make', '').upper() == 'PANASONIC'
        
        # Check if it's an RW2 file
        is_rw2 = file_ext.lower() == '.rw2'
        
        return is_panasonic and is_rw2
    
    @performance_monitor
    def extract_metadata(self, image_path: str, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract Panasonic-specific metadata
        
        Args:
            image_path: Path to the image file
            exif_data: Basic EXIF data
            
        Returns:
            Dictionary with extracted metadata
        """
        with MemoryTracker(self.memory_limit) as tracker:
            start_time = time.time()
            result = exif_data.copy()
            
            # Add file type
            result['file_type'] = 'RW2'
            
            # Define tasks to run in parallel
            def extract_exiftool_data():
                exiftool_result = {}
                if self._check_exiftool():
                    panasonic_data = self._extract_with_exiftool(image_path)
                    if panasonic_data:
                        # Add Panasonic-specific fields with panasonic_ prefix
                        for key, value in panasonic_data.items():
                            exiftool_result[f'panasonic_{key.lower()}'] = value
                        
                        # Extract Panasonic RAW version
                        if 'Panasonic Raw Version' in panasonic_data:
                            exiftool_result['panasonic_raw_version'] = panasonic_data['Panasonic Raw Version']
                        
                        # Extract sensor information
                        if 'Sensor Width' in panasonic_data and 'Sensor Height' in panasonic_data:
                            exiftool_result['panasonic_sensor_width'] = panasonic_data['Sensor Width']
                            exiftool_result['panasonic_sensor_height'] = panasonic_data['Sensor Height']
                        
                        # Extract white balance information
                        if 'WB Red Level' in panasonic_data and 'WB Green Level' in panasonic_data and 'WB Blue Level' in panasonic_data:
                            exiftool_result['panasonic_wb_red'] = panasonic_data['WB Red Level']
                            exiftool_result['panasonic_wb_green'] = panasonic_data['WB Green Level']
                            exiftool_result['panasonic_wb_blue'] = panasonic_data['WB Blue Level']
                        
                        # Extract noise reduction parameters
                        if 'Noise Reduction Params' in panasonic_data:
                            exiftool_result['panasonic_noise_reduction'] = panasonic_data['Noise Reduction Params']
                        
                        # Extract lens information
                        if 'Lens Type' in panasonic_data:
                            exiftool_result['panasonic_lens'] = panasonic_data['Lens Type']
                        
                        # Extract advanced features
                        if 'Image Stabilization' in panasonic_data:
                            exiftool_result['panasonic_stabilization'] = panasonic_data['Image Stabilization']
                        
                        if 'Shooting Mode' in panasonic_data:
                            exiftool_result['panasonic_shooting_mode'] = panasonic_data['Shooting Mode']
                        
                        if 'Advanced Scene Mode' in panasonic_data:
                            exiftool_result['panasonic_scene_mode'] = panasonic_data['Advanced Scene Mode']
                return exiftool_result
            
            def extract_model_features():
                model_result = {}
                # Extract camera model specific features
                camera_model = exif_data.get('camera_model', '')
                if camera_model in self.panasonic_models:
                    for feature, value in self.panasonic_models[camera_model].items():
                        model_result[f'panasonic_{feature}'] = value
                return model_result
            
            def extract_raw_data():
                raw_result = {}
                # Try to extract raw data using rawpy
                try:
                    with rawpy.imread(image_path) as raw:
                        # Extract raw metadata
                        raw_result['panasonic_raw_pattern'] = str(raw.raw_pattern)
                        raw_result['panasonic_color_desc'] = str(raw.color_desc)
                        
                        # Extract black levels
                        if hasattr(raw, 'black_level_per_channel'):
                            raw_result['panasonic_black_levels'] = str(raw.black_level_per_channel)
                        
                        # Extract white balance coefficients if available
                        if hasattr(raw, 'camera_whitebalance'):
                            raw_result['panasonic_camera_wb'] = str(raw.camera_whitebalance)
                        
                        # Extract raw image size
                        raw_result['panasonic_raw_width'] = raw.sizes.raw_width
                        raw_result['panasonic_raw_height'] = raw.sizes.raw_height
                        
                        # If GPU is available, use it for additional processing
                        if self.use_gpu:
                            try:
                                import torch
                                # Get raw image data
                                if hasattr(raw, 'raw_image'):
                                    # Sample the raw image to avoid memory issues
                                    raw_image = raw.raw_image
                                    sample_rate = max(1, raw_image.size // (1024*1024))  # Sample to ~1M pixels
                                    sampled_image = raw_image[::sample_rate]
                                    
                                    # Process on GPU
                                    device = torch.device("mps")
                                    tensor = torch.tensor(sampled_image, device=device)
                                    
                                    # Calculate statistics
                                    raw_result['panasonic_raw_mean'] = float(tensor.float().mean().cpu().numpy())
                                    raw_result['panasonic_raw_std'] = float(tensor.float().std().cpu().numpy())
                                    raw_result['panasonic_raw_min'] = int(tensor.min().cpu().numpy())
                                    raw_result['panasonic_raw_max'] = int(tensor.max().cpu().numpy())
                            except Exception as e:
                                logger.error(f"GPU processing error: {e}")
                except Exception as e:
                    logger.error(f"Error extracting raw data: {e}")
                return raw_result
            
            # Execute tasks in parallel
            with self.thread_pool.get_pool() as executor:
                exiftool_future = executor.submit(extract_exiftool_data)
                model_future = executor.submit(extract_model_features)
                raw_future = executor.submit(extract_raw_data)
                
                # Get results
                try:
                    exiftool_result = exiftool_future.result()
                    result.update(exiftool_result)
                except Exception as e:
                    logger.error(f"Error in exiftool extraction: {e}")
                
                try:
                    model_result = model_future.result()
                    result.update(model_result)
                except Exception as e:
                    logger.error(f"Error in model features extraction: {e}")
                
                try:
                    raw_result = raw_future.result()
                    result.update(raw_result)
                except Exception as e:
                    logger.error(f"Error in raw data extraction: {e}")
            
            # Count Panasonic-specific fields
            panasonic_fields = [key for key in result.keys() if key.startswith('panasonic_')]
            result['panasonic_field_count'] = len(panasonic_fields)
            
            # Log performance metrics
            end_time = time.time()
            memory_used, peak_memory, peak_percentage = tracker.end()
            logger.info(f"Extracted {len(panasonic_fields)} Panasonic-specific fields in {end_time - start_time:.2f} seconds")
            logger.info(f"Memory usage: {memory_used/(1024*1024):.2f} MB, Peak: {peak_memory/(1024*1024):.2f} MB ({peak_percentage:.1f}%)")
            
            return result
    
    def _extract_with_exiftool(self, image_path: str) -> Dict[str, Any]:
        """
        Extract metadata using exiftool
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with extracted metadata
        """
        try:
            cmd = ['exiftool', '-j', '-n', image_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)[0]
            return data
        except Exception as e:
            print(f"Error running exiftool: {e}")
            return {}
    
    @performance_monitor
    def process_raw(self, image_path: str, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process RAW file data for Panasonic cameras
        
        Args:
            image_path: Path to the image file
            exif_data: Basic EXIF data already extracted
            
        Returns:
            Dictionary containing the processed RAW data
        """
        with MemoryTracker(self.memory_limit) as tracker:
            start_time = time.time()
            result = {}
            
            # Define parallel tasks for raw processing
            def extract_basic_metadata(raw):
                basic_result = {}
                try:
                    # Extract raw metadata
                    basic_result['raw_pattern'] = str(raw.raw_pattern)
                    basic_result['color_desc'] = str(raw.color_desc)
                    
                    # Extract raw image size
                    basic_result['raw_width'] = raw.sizes.raw_width
                    basic_result['raw_height'] = raw.sizes.raw_height
                    
                    # Extract Panasonic-specific raw properties
                    if hasattr(raw, 'raw_type') and raw.raw_type:
                        basic_result['raw_type'] = raw.raw_type
                        
                    # Get raw image pattern
                    if hasattr(raw, 'color_desc'):
                        basic_result['raw_pattern_desc'] = raw.color_desc.decode('utf-8') if isinstance(raw.color_desc, bytes) else str(raw.color_desc)
                except Exception as e:
                    logger.error(f"Error extracting basic raw metadata: {e}")
                return basic_result
            
            def extract_levels_and_wb(raw):
                levels_result = {}
                try:
                    # Extract black levels
                    if hasattr(raw, 'black_level_per_channel'):
                        levels_result['black_levels'] = str(raw.black_level_per_channel)
                    
                    # Extract white balance coefficients if available
                    if hasattr(raw, 'camera_whitebalance'):
                        levels_result['camera_wb'] = str(raw.camera_whitebalance)
                except Exception as e:
                    logger.error(f"Error extracting levels and white balance: {e}")
                return levels_result
            
            def extract_advanced_data(raw):
                advanced_result = {}
                try:
                    # Only process if we have GPU support and it's enabled
                    if self.use_gpu:
                        try:
                            import torch
                            # Process raw data on GPU for advanced analysis
                            if hasattr(raw, 'raw_image'):
                                # Use safe array operation to prevent memory issues
                                with safe_array_operation():
                                    # Sample the raw image to avoid memory issues
                                    raw_image = raw.raw_image
                                    sample_rate = max(1, raw_image.size // (1024*1024))  # Sample to ~1M pixels
                                    sampled_image = raw_image[::sample_rate]
                                    
                                    # Process on GPU (Apple Silicon MPS)
                                    device = torch.device("mps")
                                    tensor = torch.tensor(sampled_image, device=device)
                                    
                                    # Calculate advanced statistics
                                    advanced_result['raw_mean'] = float(tensor.float().mean().cpu().numpy())
                                    advanced_result['raw_std'] = float(tensor.float().std().cpu().numpy())
                                    advanced_result['raw_min'] = int(tensor.min().cpu().numpy())
                                    advanced_result['raw_max'] = int(tensor.max().cpu().numpy())
                                    
                                    # Calculate histogram for exposure analysis
                                    hist = torch.histc(tensor.float(), bins=256, min=0, max=65535)
                                    hist_np = hist.cpu().numpy()
                                    
                                    # Find peaks in histogram (simplified)
                                    peaks = [i for i in range(1, 255) if hist_np[i] > hist_np[i-1] and hist_np[i] > hist_np[i+1]]
                                    if peaks:
                                        advanced_result['histogram_peaks'] = str(peaks)
                                    
                                    # Detect potential highlight clipping
                                    highlight_threshold = 65000  # Close to max value
                                    highlight_percentage = (tensor > highlight_threshold).sum().item() / tensor.numel() * 100
                                    advanced_result['highlight_clipping_percentage'] = round(highlight_percentage, 2)
                        except Exception as e:
                            logger.error(f"Error in GPU-accelerated processing: {e}")
                except Exception as e:
                    logger.error(f"Error extracting advanced data: {e}")
                return advanced_result
            
            try:
                # Open the raw file once and process in parallel
                with rawpy.imread(image_path) as raw:
                    # Use thread pool for parallel processing
                    with self.thread_pool.get_pool() as executor:
                        # Submit tasks
                        basic_future = executor.submit(extract_basic_metadata, raw)
                        levels_future = executor.submit(extract_levels_and_wb, raw)
                        advanced_future = executor.submit(extract_advanced_data, raw)
                        
                        # Collect results
                        try:
                            basic_result = basic_future.result()
                            result.update(basic_result)
                        except Exception as e:
                            logger.error(f"Error getting basic metadata results: {e}")
                        
                        try:
                            levels_result = levels_future.result()
                            result.update(levels_result)
                        except Exception as e:
                            logger.error(f"Error getting levels and WB results: {e}")
                        
                        try:
                            advanced_result = advanced_future.result()
                            result.update(advanced_result)
                        except Exception as e:
                            logger.error(f"Error getting advanced data results: {e}")
            except Exception as e:
                logger.error(f"Error processing Panasonic RAW data: {e}")
            
            # Log performance metrics
            end_time = time.time()
            memory_used, peak_memory, peak_percentage = tracker.end()
            logger.info(f"Processed RAW data with {len(result)} attributes in {end_time - start_time:.2f} seconds")
            logger.info(f"Memory usage: {memory_used/(1024*1024):.2f} MB, Peak: {peak_memory/(1024*1024):.2f} MB ({peak_percentage:.1f}%)")
            
            return result
    
    def get_makernote_tags(self) -> Dict[str, str]:
        """
        Get MakerNote tag mapping for Panasonic cameras
        
        Returns:
            Dictionary mapping MakerNote tag names to field names
        """
        return {
            # Panasonic MakerNote tags
            'PanasonicRawVersion': 'panasonic_raw_version',
            'SensorWidth': 'panasonic_sensor_width',
            'SensorHeight': 'panasonic_sensor_height',
            'SensorTopBorder': 'panasonic_sensor_top_border',
            'SensorLeftBorder': 'panasonic_sensor_left_border',
            'SensorBottomBorder': 'panasonic_sensor_bottom_border',
            'SensorRightBorder': 'panasonic_sensor_right_border',
            'BlackLevelRed': 'panasonic_black_level_red',
            'BlackLevelGreen': 'panasonic_black_level_green',
            'BlackLevelBlue': 'panasonic_black_level_blue',
            'WBRedLevel': 'panasonic_wb_red_level',
            'WBGreenLevel': 'panasonic_wb_green_level',
            'WBBlueLevel': 'panasonic_wb_blue_level',
            'ISO': 'panasonic_iso',
            'NoiseReductionParams': 'panasonic_noise_reduction_params',
            'ImageStabilization': 'panasonic_stabilization',
            'ShootingMode': 'panasonic_shooting_mode',
            'AdvancedSceneMode': 'panasonic_scene_mode',
            'FocusMode': 'panasonic_focus_mode',
            'AFAreaMode': 'panasonic_af_area_mode',
            'MacroMode': 'panasonic_macro_mode',
            'BurstMode': 'panasonic_burst_mode',
            'WhiteBalance': 'panasonic_white_balance',
            'LensType': 'panasonic_lens_type',
            'LensTypeMake': 'panasonic_lens_make',
            'LensTypeModel': 'panasonic_lens_model',
        }
    
    def _check_exiftool(self) -> bool:
        """
        Check if exiftool is available
        
        Returns:
            True if exiftool is available, False otherwise
        """
        try:
            subprocess.run(['exiftool', '-ver'], capture_output=True, check=True)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            print("Warning: exiftool not found, some metadata extraction will be limited")
            return False
