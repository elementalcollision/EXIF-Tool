#!/usr/bin/env python3
"""
Nikon Camera Extractor
Provides Nikon-specific EXIF extraction for NEF files
"""

import os
import io
import time
import logging
import numpy as np
import rawpy
from PIL import Image
from typing import Dict, Any, Optional, List, Tuple, Callable

from .base_extractor import CameraExtractor
from camera_extractors.optimization_utils import (
    MemoryTracker, performance_monitor, safe_array_operation
)

# Configure logging
logger = logging.getLogger('nikon_extractor')


class NikonExtractor(CameraExtractor):
    """Nikon-specific EXIF extractor for NEF files"""
    
    def __init__(self, use_gpu=False, memory_limit=0.75, cpu_cores=None):
        super().__init__(use_gpu=use_gpu, memory_limit=memory_limit, cpu_cores=cpu_cores)
        
        # Log initialization
        logger.info(f"Initializing NikonExtractor with GPU={use_gpu}, "
                  f"memory_limit={memory_limit}, cpu_cores={cpu_cores}")
    
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
    
    @performance_monitor
    def extract_metadata(self, image_path: str, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Nikon-specific metadata from the image
        
        Args:
            image_path: Path to the image file
            exif_data: Basic EXIF data already extracted
            
        Returns:
            Dictionary containing Nikon-specific metadata
        """
        with MemoryTracker(self.memory_limit) as tracker:
            start_time = time.time()
            result = {}
            
            # Define tasks to run in parallel
            def process_makernote_tags():
                tag_result = {}
                # If there are Nikon MakerNote tags, process them
                nikon_tags = {k: v for k, v in exif_data.items() if k.startswith('nikon_')}
                if nikon_tags:
                    tag_result['has_nikon_makernote'] = True
                    tag_result['nikon_makernote_count'] = len(nikon_tags)
                    
                    # Process specific Nikon tags if present
                    if 'nikon_lens_type' in nikon_tags:
                        tag_result['nikon_lens_identified'] = True
                    
                    # Group tags by category for easier analysis
                    lens_tags = {k: v for k, v in nikon_tags.items() if 'lens' in k.lower()}
                    if lens_tags:
                        tag_result['nikon_lens_tag_count'] = len(lens_tags)
                    
                    focus_tags = {k: v for k, v in nikon_tags.items() if 'focus' in k.lower()}
                    if focus_tags:
                        tag_result['nikon_focus_tag_count'] = len(focus_tags)
                        
                    # Check for advanced features
                    if 'nikon_active_d_lighting' in nikon_tags:
                        tag_result['nikon_has_active_d_lighting'] = True
                        
                    if 'nikon_vr_info' in nikon_tags:
                        tag_result['nikon_has_vibration_reduction'] = True
                return tag_result
            
            def extract_model_features():
                model_result = {}
                # Add Nikon camera model specific features
                camera_model = exif_data.get('camera_model', '')
                
                # Detect Nikon camera series
                if 'Z' in camera_model:
                    model_result['nikon_camera_series'] = 'Z-series'
                    model_result['nikon_mount_type'] = 'Z-mount'
                elif 'D' in camera_model:
                    model_result['nikon_camera_series'] = 'D-series'
                    model_result['nikon_mount_type'] = 'F-mount'
                
                # Detect specific camera features based on model
                if any(x in camera_model for x in ['D850', 'D810', 'Z7']):
                    model_result['nikon_high_resolution'] = True
                if any(x in camera_model for x in ['D5', 'D6', 'D500', 'Z9']):
                    model_result['nikon_professional'] = True
                if any(x in camera_model for x in ['D750', 'D780', 'Z6']):
                    model_result['nikon_all_around'] = True
                return model_result
            
            def extract_file_metadata():
                file_result = {}
                # Add file type
                file_result['file_type'] = 'NEF'
                
                # Get file size
                try:
                    file_result['file_size'] = os.path.getsize(image_path)
                except Exception as e:
                    logger.error(f"Error getting file size: {e}")
                
                return file_result
            
            # Execute tasks in parallel
            with self.thread_pool.get_pool() as executor:
                makernote_future = executor.submit(process_makernote_tags)
                model_future = executor.submit(extract_model_features)
                file_future = executor.submit(extract_file_metadata)
                
                # Get results
                try:
                    makernote_result = makernote_future.result()
                    result.update(makernote_result)
                except Exception as e:
                    logger.error(f"Error in MakerNote extraction: {e}")
                
                try:
                    model_result = model_future.result()
                    result.update(model_result)
                except Exception as e:
                    logger.error(f"Error in model features extraction: {e}")
                
                try:
                    file_result = file_future.result()
                    result.update(file_result)
                except Exception as e:
                    logger.error(f"Error in file metadata extraction: {e}")
            
            # Count Nikon-specific fields
            nikon_fields = [key for key in result.keys() if key.startswith('nikon_')]
            result['nikon_field_count'] = len(nikon_fields)
            
            # Log performance metrics
            end_time = time.time()
            memory_used, peak_memory, peak_percentage = tracker.end()
            logger.info(f"Extracted {len(nikon_fields)} Nikon-specific fields in {end_time - start_time:.2f} seconds")
            logger.info(f"Memory usage: {memory_used/(1024*1024):.2f} MB, Peak: {peak_memory/(1024*1024):.2f} MB ({peak_percentage:.1f}%)")
            
            return result
    
    @performance_monitor
    def process_raw(self, image_path: str, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Nikon NEF file data
        
        Args:
            image_path: Path to the NEF file
            exif_data: Basic EXIF data already extracted
            
        Returns:
            Dictionary containing processed Nikon NEF data
        """
        with MemoryTracker(self.memory_limit) as tracker:
            start_time = time.time()
            result = {}
            
            # Define parallel tasks for raw processing
            def extract_basic_metadata(raw):
                basic_result = {}
                try:
                    # Nikon NEF version detection
                    basic_result['nikon_nef_version'] = 'NEF 1.0' if str(raw.raw_type) == 'RawType.Flat' else 'NEF 2.0'
                    
                    # Try to get color profile information
                    if hasattr(raw, 'color_desc'):
                        basic_result['nikon_color_profile'] = raw.color_desc.decode('utf-8', errors='ignore')
                        logger.info(f"Color profile: {basic_result['nikon_color_profile']}")
                    
                    # Get white balance coefficients if available
                    if hasattr(raw, 'camera_whitebalance'):
                        # Handle different types of camera_whitebalance
                        if hasattr(raw.camera_whitebalance, 'tolist'):
                            basic_result['nikon_camera_whitebalance'] = raw.camera_whitebalance.tolist()
                        else:
                            basic_result['nikon_camera_whitebalance'] = str(raw.camera_whitebalance)
                    
                    # Get full resolution dimensions
                    if hasattr(raw, 'sizes'):
                        basic_result['nikon_full_width'] = raw.sizes.width
                        basic_result['nikon_full_height'] = raw.sizes.height
                        basic_result['nikon_raw_width'] = raw.sizes.raw_width
                        basic_result['nikon_raw_height'] = raw.sizes.raw_height
                        
                    # Get black and white levels
                    basic_result['nikon_raw_black_level'] = int(raw.black_level) if hasattr(raw, 'black_level') else 0
                    basic_result['nikon_raw_white_level'] = int(raw.white_level) if hasattr(raw, 'white_level') else 0
                except Exception as e:
                    logger.error(f"Error extracting basic raw metadata: {e}")
                return basic_result
            
            def extract_raw_stats(raw):
                stats_result = {}
                try:
                    with safe_array_operation():
                        # Get raw image data - use sampling to reduce memory usage
                        raw_image = raw.raw_image
                        
                        # For large images, sample to reduce memory usage
                        if raw_image.size > 20_000_000:  # For very large sensors
                            sample_rate = max(1, int(np.sqrt(raw_image.size / 1_000_000)))
                            sampled_image = raw_image[::sample_rate, ::sample_rate]
                            logger.info(f"Sampling raw image at 1/{sample_rate} for stats calculation")
                        else:
                            sampled_image = raw_image
                        
                        # Get basic stats about the raw image
                        raw_min = np.min(sampled_image)
                        raw_max = np.max(sampled_image)
                        
                        # Record shape and min/max
                        stats_result['nikon_raw_image_shape'] = str(raw_image.shape)
                        stats_result['nikon_raw_min_value'] = int(raw_min)
                        stats_result['nikon_raw_max_value'] = int(raw_max)
                        
                        # Calculate histogram with reduced bins for efficiency
                        histogram, _ = np.histogram(sampled_image.flatten(), bins=256)
                        
                        # Calculate basic statistics
                        stats_result['nikon_raw_histogram_mean'] = float(np.mean(histogram))
                        stats_result['nikon_raw_histogram_std'] = float(np.std(histogram))
                        stats_result['nikon_raw_dynamic_range'] = float(np.log2(raw_max - raw_min + 1)) if raw_max > raw_min else 0
                        
                        # If GPU is available, do more advanced processing
                        if self.use_gpu:
                            try:
                                import torch
                                if torch.backends.mps.is_available():
                                    # Convert to tensor and move to GPU
                                    device = torch.device("mps")
                                    tensor = torch.tensor(sampled_image, device=device)
                                    
                                    # Calculate percentiles for exposure analysis
                                    percentiles = [1, 5, 10, 50, 90, 95, 99]
                                    for p in percentiles:
                                        stats_result[f'nikon_raw_percentile_{p}'] = float(torch.quantile(tensor.float(), p/100).cpu().numpy())
                                    
                                    # Calculate exposure metrics
                                    highlight_threshold = 0.95 * raw_max
                                    shadow_threshold = raw_min + 0.05 * (raw_max - raw_min)
                                    
                                    highlight_percentage = (tensor > highlight_threshold).sum().item() / tensor.numel() * 100
                                    shadow_percentage = (tensor < shadow_threshold).sum().item() / tensor.numel() * 100
                                    
                                    stats_result['nikon_highlight_percentage'] = round(highlight_percentage, 2)
                                    stats_result['nikon_shadow_percentage'] = round(shadow_percentage, 2)
                                    stats_result['nikon_midtone_percentage'] = round(100 - highlight_percentage - shadow_percentage, 2)
                            except Exception as e:
                                logger.error(f"GPU processing error: {e}")
                except Exception as e:
                    logger.error(f"Error extracting raw stats: {e}")
                return stats_result
            
            def extract_thumbnail(raw):
                thumb_result = {}
                try:
                    thumb = raw.extract_thumb()
                    if thumb and hasattr(thumb, 'format'):
                        thumb_result['has_thumbnail'] = True
                        thumb_result['thumbnail_format'] = thumb.format
                        logger.info(f"Extracted thumbnail in {thumb.format} format")
                        
                        # Process thumbnail if needed
                        if thumb.format == 'jpeg':
                            thumbnail_data = self.process_thumbnail(thumb.data, thumb.format)
                            if thumbnail_data:
                                thumb_result.update(thumbnail_data)
                except Exception as e:
                    thumb_result['has_thumbnail'] = False
                    logger.error(f"Thumbnail extraction error: {e}")
                return thumb_result
            
            try:
                # Open the raw file once and process in parallel
                with rawpy.imread(image_path) as raw:
                    logger.info(f"Processing Nikon NEF file: {image_path}")
                    
                    # Use thread pool for parallel processing
                    with self.thread_pool.get_pool() as executor:
                        # Submit tasks
                        basic_future = executor.submit(extract_basic_metadata, raw)
                        stats_future = executor.submit(extract_raw_stats, raw)
                        thumb_future = executor.submit(extract_thumbnail, raw)
                        
                        # Collect results
                        try:
                            basic_result = basic_future.result()
                            result.update(basic_result)
                        except Exception as e:
                            logger.error(f"Error getting basic metadata results: {e}")
                        
                        try:
                            stats_result = stats_future.result()
                            result.update(stats_result)
                        except Exception as e:
                            logger.error(f"Error getting raw stats results: {e}")
                        
                        try:
                            thumb_result = thumb_future.result()
                            result.update(thumb_result)
                        except Exception as e:
                            logger.error(f"Error getting thumbnail results: {e}")
            except Exception as e:
                logger.error(f"Nikon NEF processing error: {e}")
            
            # Log performance metrics
            end_time = time.time()
            memory_used, peak_memory, peak_percentage = tracker.end()
            logger.info(f"Processed Nikon NEF data with {len(result)} attributes in {end_time - start_time:.2f} seconds")
            logger.info(f"Memory usage: {memory_used/(1024*1024):.2f} MB, Peak: {peak_memory/(1024*1024):.2f} MB ({peak_percentage:.1f}%)")
            
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
    
    @performance_monitor
    def process_thumbnail(self, thumb_data: bytes, thumb_format: str) -> Optional[Dict[str, Any]]:
        """Process Nikon thumbnail data with GPU acceleration if available
        
        Args:
            thumb_data: Raw thumbnail data
            thumb_format: Format of the thumbnail (e.g., 'jpeg')
            
        Returns:
            Dictionary containing processed thumbnail data, or None
        """
        with MemoryTracker(self.memory_limit) as tracker:
            start_time = time.time()
            result = {}
            
            try:
                # Basic thumbnail processing
                with Image.open(io.BytesIO(thumb_data)) as pil_thumb:
                    # Get thumbnail dimensions
                    thumb_width, thumb_height = pil_thumb.size
                    result['thumbnail_width'] = thumb_width
                    result['thumbnail_height'] = thumb_height
                    result['thumbnail_aspect_ratio'] = round(thumb_width / thumb_height, 2)
                    
                    # If using GPU acceleration, do more advanced processing
                    if self.use_gpu and thumb_format == 'jpeg':
                        try:
                            import torch
                            if torch.backends.mps.is_available():
                                logger.info("Processing thumbnail with Metal GPU acceleration")
                                
                                # Convert to numpy array
                                img_array = np.array(pil_thumb)
                                
                                # Move to GPU
                                device = torch.device("mps")
                                tensor = torch.tensor(img_array, device=device)
                                
                                # Calculate color statistics
                                if len(tensor.shape) == 3 and tensor.shape[2] >= 3:
                                    # RGB channels
                                    r_mean = float(tensor[:,:,0].float().mean().cpu().numpy())
                                    g_mean = float(tensor[:,:,1].float().mean().cpu().numpy())
                                    b_mean = float(tensor[:,:,2].float().mean().cpu().numpy())
                                    
                                    result['thumbnail_r_mean'] = round(r_mean, 2)
                                    result['thumbnail_g_mean'] = round(g_mean, 2)
                                    result['thumbnail_b_mean'] = round(b_mean, 2)
                                    
                                    # Calculate color balance
                                    rg_ratio = round(r_mean / g_mean, 2) if g_mean > 0 else 0
                                    rb_ratio = round(r_mean / b_mean, 2) if b_mean > 0 else 0
                                    gb_ratio = round(g_mean / b_mean, 2) if b_mean > 0 else 0
                                    
                                    result['thumbnail_rg_ratio'] = rg_ratio
                                    result['thumbnail_rb_ratio'] = rb_ratio
                                    result['thumbnail_gb_ratio'] = gb_ratio
                                    
                                    # Estimate color temperature (simplified)
                                    if rb_ratio > 1.1:
                                        result['thumbnail_color_temp_estimate'] = 'Warm'
                                    elif rb_ratio < 0.9:
                                        result['thumbnail_color_temp_estimate'] = 'Cool'
                                    else:
                                        result['thumbnail_color_temp_estimate'] = 'Neutral'
                        except Exception as e:
                            logger.error(f"GPU thumbnail processing error: {e}")
            except Exception as e:
                logger.error(f"Thumbnail processing error: {e}")
            
            # Log performance metrics
            end_time = time.time()
            memory_used, peak_memory, peak_percentage = tracker.end()
            logger.info(f"Processed thumbnail with {len(result)} attributes in {end_time - start_time:.2f} seconds")
            
            return result
