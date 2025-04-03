#!/usr/bin/env python3
"""
Fujifilm Camera Extractor
Provides Fujifilm-specific EXIF extraction for RAF files
"""

import os
import io
import time
import logging
import subprocess
import json
import tempfile
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, List, Tuple, Callable

from .base_extractor import CameraExtractor
from camera_extractors.optimization_utils import (
    MemoryTracker, performance_monitor, safe_array_operation
)

# Configure logging
logger = logging.getLogger('fujifilm_extractor')


class FujifilmExtractor(CameraExtractor):
    """Fujifilm-specific EXIF extractor for RAF files"""
    
    def __init__(self, use_gpu=False, memory_limit=0.75, cpu_cores=None):
        super().__init__(use_gpu=use_gpu, memory_limit=memory_limit, cpu_cores=cpu_cores)
        
        # Log initialization
        logger.info(f"Initializing FujifilmExtractor with GPU={use_gpu}, "
                  f"memory_limit={memory_limit}, cpu_cores={cpu_cores}")
    
    def can_handle(self, file_ext: str, exif_data: Dict[str, Any]) -> bool:
        """Check if this extractor can handle the given file
        
        Args:
            file_ext: File extension (e.g., '.raf')
            exif_data: Basic EXIF data already extracted
            
        Returns:
            True if this is a Fujifilm RAF file, False otherwise
        """
        # Check file extension
        if file_ext.lower() != '.raf':
            return False
            
        # Check camera make if available
        is_fujifilm = any(x in exif_data.get('camera_make', '').upper() for x in ['FUJI', 'FUJIFILM'])
        
        # If camera_make is not available but it's a RAF file, assume it's Fujifilm
        if 'camera_make' not in exif_data:
            return True
            
        return is_fujifilm
    
    @performance_monitor
    def extract_metadata(self, image_path: str, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Fujifilm-specific metadata from the image
        
        Args:
            image_path: Path to the image file
            exif_data: Basic EXIF data already extracted
            
        Returns:
            Dictionary containing Fujifilm-specific metadata
        """
        with MemoryTracker(self.memory_limit) as tracker:
            start_time = time.time()
            result = {}
            
            # Define tasks to run in parallel
            def extract_exiftool_data():
                exiftool_result = {}
                try:
                    if self._check_exiftool():
                        logger.info("Using exiftool to extract Fujifilm metadata")
                        exiftool_data = self._run_exiftool(image_path)
                        if exiftool_data:
                            # Process Fujifilm-specific tags
                            for key, value in exiftool_data.items():
                                if key.startswith('Fuji'):
                                    # Convert to snake_case
                                    fuji_key = 'fuji_' + key[4:].lower().replace(' ', '_')
                                    exiftool_result[fuji_key] = value
                            
                            # Extract important Fujifilm metadata
                            if 'Make' in exiftool_data:
                                exiftool_result['camera_make'] = exiftool_data['Make']
                            if 'Model' in exiftool_data:
                                exiftool_result['camera_model'] = exiftool_data['Model']
                            if 'DateTimeOriginal' in exiftool_data:
                                exiftool_result['date_taken'] = exiftool_data['DateTimeOriginal']
                            if 'ExposureTime' in exiftool_data:
                                exiftool_result['exposure_time'] = exiftool_data['ExposureTime']
                            if 'FNumber' in exiftool_data:
                                exiftool_result['f_number'] = exiftool_data['FNumber']
                            if 'ISO' in exiftool_data:
                                exiftool_result['iso'] = exiftool_data['ISO']
                            if 'FocalLength' in exiftool_data:
                                exiftool_result['focal_length'] = exiftool_data['FocalLength']
                            
                            # Extract Fujifilm-specific features
                            fuji_specific = {
                                'FujiFilmVersion': 'fuji_version',
                                'FujiFilmSerialNumber': 'fuji_serial_number',
                                'FujiFilmQuality': 'fuji_quality',
                                'FujiFilmSharpness': 'fuji_sharpness',
                                'FujiFilmWhiteBalance': 'fuji_white_balance',
                                'FujiFilmColorMode': 'fuji_color_mode',
                                'FujiFilmToneMode': 'fuji_tone_mode',
                                'FujiFilmDynamicRange': 'fuji_dynamic_range',
                                'FujiFilmFilmMode': 'fuji_film_simulation',
                                'FujiFilmFinePixColor': 'fuji_fine_pix_color',
                                'FujiFilmFocusMode': 'fuji_focus_mode',
                                'FujiFilmFocusArea': 'fuji_focus_area',
                                'FujiFilmFocusPoint': 'fuji_focus_point',
                                'FujiFilmSlowSync': 'fuji_slow_sync',
                                'FujiFilmFlashMode': 'fuji_flash_mode',
                                'FujiFilmExrMode': 'fuji_exr_mode',
                                'FujiFilmShutterType': 'fuji_shutter_type',
                                'FujiFilmContinuousShooting': 'fuji_continuous_shooting',
                                'FujiFilmSequenceNumber': 'fuji_sequence_number'
                            }
                            
                            for exif_key, result_key in fuji_specific.items():
                                if exif_key in exiftool_data:
                                    exiftool_result[result_key] = exiftool_data[exif_key]
                except Exception as e:
                    logger.error(f"Error extracting Fujifilm metadata with exiftool: {e}")
                return exiftool_result
            
            def extract_model_features():
                model_result = {}
                # Add Fujifilm camera model specific features
                camera_model = exif_data.get('camera_model', '')
                
                # Detect Fujifilm camera series
                if 'X-T' in camera_model:
                    model_result['fuji_camera_series'] = 'X-T'
                    model_result['fuji_mirrorless'] = True
                    if any(x in camera_model for x in ['X-T3', 'X-T4', 'X-T5']):
                        model_result['fuji_x_trans4'] = True
                elif 'X-Pro' in camera_model:
                    model_result['fuji_camera_series'] = 'X-Pro'
                    model_result['fuji_mirrorless'] = True
                    model_result['fuji_rangefinder_style'] = True
                elif 'X-E' in camera_model:
                    model_result['fuji_camera_series'] = 'X-E'
                    model_result['fuji_mirrorless'] = True
                elif 'X-H' in camera_model:
                    model_result['fuji_camera_series'] = 'X-H'
                    model_result['fuji_mirrorless'] = True
                    model_result['fuji_ibis'] = True
                elif 'GFX' in camera_model:
                    model_result['fuji_camera_series'] = 'GFX'
                    model_result['fuji_medium_format'] = True
                
                return model_result
            
            def extract_file_metadata():
                file_result = {}
                # Add file type
                file_result['file_type'] = 'RAF'
                
                # Get file size
                try:
                    file_result['file_size'] = os.path.getsize(image_path)
                except Exception as e:
                    logger.error(f"Error getting file size: {e}")
                
                return file_result
            
            # Execute tasks in parallel
            with self.thread_pool.get_pool() as executor:
                exiftool_future = executor.submit(extract_exiftool_data)
                model_future = executor.submit(extract_model_features)
                file_future = executor.submit(extract_file_metadata)
                
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
                    file_result = file_future.result()
                    result.update(file_result)
                except Exception as e:
                    logger.error(f"Error in file metadata extraction: {e}")
            
            # Count Fujifilm-specific fields
            fuji_fields = [key for key in result.keys() if key.startswith('fuji_')]
            result['fuji_field_count'] = len(fuji_fields)
            
            # Log performance metrics
            end_time = time.time()
            memory_used, peak_memory, peak_percentage = tracker.end()
            logger.info(f"Extracted {len(fuji_fields)} Fujifilm-specific fields in {end_time - start_time:.2f} seconds")
            logger.info(f"Memory usage: {memory_used/(1024*1024):.2f} MB, Peak: {peak_memory/(1024*1024):.2f} MB ({peak_percentage:.1f}%)")
            
            return result
    
    @performance_monitor
    def process_raw(self, image_path: str, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Fujifilm RAW file data
        
        Args:
            image_path: Path to the RAF file
            exif_data: Basic EXIF data already extracted
            
        Returns:
            Dictionary containing processed Fujifilm RAW data
        """
        with MemoryTracker(self.memory_limit) as tracker:
            start_time = time.time()
            result = {}
            
            logger.info(f"Processing Fujifilm RAF specific data")
            
            # Define parallel tasks for RAF processing
            def extract_basic_metadata(raw):
                basic_result = {}
                try:
                    # Get basic metadata
                    if hasattr(raw, 'color_desc'):
                        basic_result['fuji_color_profile'] = raw.color_desc.decode('utf-8', errors='ignore')
                    
                    # Get white balance coefficients if available
                    if hasattr(raw, 'camera_whitebalance'):
                        if hasattr(raw.camera_whitebalance, 'tolist'):
                            basic_result['fuji_camera_whitebalance'] = raw.camera_whitebalance.tolist()
                        else:
                            basic_result['fuji_camera_whitebalance'] = str(raw.camera_whitebalance)
                    
                    # Get image dimensions
                    if hasattr(raw, 'sizes'):
                        basic_result['fuji_full_width'] = raw.sizes.width
                        basic_result['fuji_full_height'] = raw.sizes.height
                        basic_result['fuji_raw_width'] = raw.sizes.raw_width
                        basic_result['fuji_raw_height'] = raw.sizes.raw_height
                    
                    # Get black and white levels
                    basic_result['fuji_raw_black_level'] = int(raw.black_level) if hasattr(raw, 'black_level') else 0
                    basic_result['fuji_raw_white_level'] = int(raw.white_level) if hasattr(raw, 'white_level') else 0
                    
                    # Check for X-Trans sensor
                    if hasattr(raw, 'raw_pattern'):
                        if hasattr(raw.raw_pattern, 'shape') and raw.raw_pattern.shape == (6, 6):
                            basic_result['fuji_x_trans_sensor'] = True
                        else:
                            basic_result['fuji_x_trans_sensor'] = False
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
                        stats_result['fuji_raw_image_shape'] = str(raw_image.shape)
                        stats_result['fuji_raw_min_value'] = int(raw_min)
                        stats_result['fuji_raw_max_value'] = int(raw_max)
                        
                        # Calculate histogram with reduced bins for efficiency
                        histogram, _ = np.histogram(sampled_image.flatten(), bins=256)
                        
                        # Calculate basic statistics
                        stats_result['fuji_raw_histogram_mean'] = float(np.mean(histogram))
                        stats_result['fuji_raw_histogram_std'] = float(np.std(histogram))
                        stats_result['fuji_raw_dynamic_range'] = float(np.log2(raw_max - raw_min + 1)) if raw_max > raw_min else 0
                        
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
                                        stats_result[f'fuji_raw_percentile_{p}'] = float(torch.quantile(tensor.float(), p/100).cpu().numpy())
                                    
                                    # Calculate exposure metrics
                                    highlight_threshold = 0.95 * raw_max
                                    shadow_threshold = raw_min + 0.05 * (raw_max - raw_min)
                                    
                                    highlight_percentage = (tensor > highlight_threshold).sum().item() / tensor.numel() * 100
                                    shadow_percentage = (tensor < shadow_threshold).sum().item() / tensor.numel() * 100
                                    
                                    stats_result['fuji_highlight_percentage'] = round(highlight_percentage, 2)
                                    stats_result['fuji_shadow_percentage'] = round(shadow_percentage, 2)
                                    stats_result['fuji_midtone_percentage'] = round(100 - highlight_percentage - shadow_percentage, 2)
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
            
            # Process RAF file with rawpy
            try:
                import rawpy
                with rawpy.imread(image_path) as raw:
                    logger.info(f"Processing Fujifilm RAF file: {image_path}")
                    
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
            except ImportError:
                logger.error("rawpy not available for RAF processing")
            except Exception as e:
                logger.error(f"Fujifilm RAF processing error: {e}")
            
            # Log performance metrics
            end_time = time.time()
            memory_used, peak_memory, peak_percentage = tracker.end()
            logger.info(f"Processed Fujifilm RAF data with {len(result)} attributes in {end_time - start_time:.2f} seconds")
            logger.info(f"Memory usage: {memory_used/(1024*1024):.2f} MB, Peak: {peak_memory/(1024*1024):.2f} MB ({peak_percentage:.1f}%)")
            
            return result
    
    def get_makernote_tags(self) -> Dict[str, str]:
        """Get Fujifilm MakerNote tag mapping
        
        Returns:
            Dictionary mapping Fujifilm MakerNote tags to human-readable names
        """
        return {
            "0x0000": "Version",
            "0x1000": "Quality",
            "0x1001": "Sharpness",
            "0x1002": "WhiteBalance",
            "0x1003": "Color",
            "0x1004": "Tone",
            "0x1010": "FlashMode",
            "0x1011": "FlashStrength",
            "0x1020": "Macro",
            "0x1021": "FocusMode",
            "0x1022": "AFPointMode",
            "0x1023": "FocusPixel",
            "0x1030": "SlowSync",
            "0x1031": "PictureMode",
            "0x1032": "ExposureCount",
            "0x1100": "SequenceNumber",
            "0x1101": "FujiIFD",
            "0x1210": "ColorMode",
            "0x1300": "BlurWarning",
            "0x1301": "FocusWarning",
            "0x1302": "ExposureWarning",
            "0x1400": "DynamicRange",
            "0x1401": "FilmMode",
            "0x1402": "DynamicRangeSetting",
            "0x1403": "DevelopmentDynamicRange",
            "0x1404": "MinFocalLength",
            "0x1405": "MaxFocalLength",
            "0x1406": "MaxApertureAtMinFocal",
            "0x1407": "MaxApertureAtMaxFocal",
            "0x1422": "ImageStabilization",
            "0x1431": "Rating",
            "0x1436": "ImageGeneration",
            "0x1438": "ImageCount",
            "0x1441": "FacesDetected",
            "0x1443": "FacePositions",
            "0x1444": "FaceRecInfo",
            "0x1445": "FileSource",
            "0x1446": "OrderNumber",
            "0x1447": "FrameNumber"
        }
    
    def process_thumbnail(self, thumb_data, thumb_format):
        """Process thumbnail data
        
        Args:
            thumb_data: Thumbnail data
            thumb_format: Thumbnail format (e.g., 'jpeg')
            
        Returns:
            Dictionary containing processed thumbnail data
        """
        result = {}
        
        try:
            # Process thumbnail with PIL
            with io.BytesIO(thumb_data) as thumb_io:
                with Image.open(thumb_io) as thumb_img:
                    # Get thumbnail dimensions
                    width, height = thumb_img.size
                    result['thumbnail_width'] = width
                    result['thumbnail_height'] = height
                    
                    # Get thumbnail format
                    result['thumbnail_format'] = thumb_img.format
                    
                    # Process with GPU if available
                    if self.use_gpu:
                        try:
                            import torch
                            from torchvision import transforms
                            
                            if torch.backends.mps.is_available():
                                # Convert to tensor and move to GPU
                                device = torch.device("mps")
                                
                                # Convert image to tensor
                                to_tensor = transforms.ToTensor()
                                img_tensor = to_tensor(thumb_img).to(device)
                                
                                # Calculate basic stats
                                result['thumbnail_mean_r'] = float(torch.mean(img_tensor[0]).item())
                                result['thumbnail_mean_g'] = float(torch.mean(img_tensor[1]).item())
                                result['thumbnail_mean_b'] = float(torch.mean(img_tensor[2]).item())
                                
                                # Calculate standard deviation
                                result['thumbnail_std_r'] = float(torch.std(img_tensor[0]).item())
                                result['thumbnail_std_g'] = float(torch.std(img_tensor[1]).item())
                                result['thumbnail_std_b'] = float(torch.std(img_tensor[2]).item())
                        except ImportError:
                            logger.warning("GPU processing requested but torch/torchvision not available")
                        except Exception as e:
                            logger.error(f"Error in GPU thumbnail processing: {e}")
        except Exception as e:
            logger.error(f"Error processing thumbnail: {e}")
        
        return result
