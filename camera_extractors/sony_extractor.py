#!/usr/bin/env python3
"""
Sony Camera Extractor
Provides Sony-specific EXIF extraction for ARW files
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
logger = logging.getLogger('sony_extractor')


class SonyExtractor(CameraExtractor):
    """Sony-specific EXIF extractor for ARW files"""
    
    def __init__(self, use_gpu=False, memory_limit=0.75, cpu_cores=None):
        super().__init__(use_gpu=use_gpu, memory_limit=memory_limit, cpu_cores=cpu_cores)
        
        # Log initialization
        logger.info(f"Initializing SonyExtractor with GPU={use_gpu}, "
                  f"memory_limit={memory_limit}, cpu_cores={cpu_cores}")
    
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
    
    @performance_monitor
    def extract_metadata(self, image_path: str, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Sony-specific metadata from the image
        
        Args:
            image_path: Path to the image file
            exif_data: Basic EXIF data already extracted
            
        Returns:
            Dictionary containing Sony-specific metadata
        """
        with MemoryTracker(self.memory_limit) as tracker:
            start_time = time.time()
            result = {}
            
            # Define tasks to run in parallel
            def process_makernote_tags():
                tag_result = {}
                # If there are Sony MakerNote tags, process them
                sony_tags = {k: v for k, v in exif_data.items() if k.startswith('sony_')}
                if sony_tags:
                    tag_result['has_sony_makernote'] = True
                    tag_result['sony_makernote_count'] = len(sony_tags)
                    
                    # Process specific Sony tags if present
                    if 'sony_lens_id' in sony_tags:
                        tag_result['sony_lens_identified'] = True
                        # Could add lens lookup functionality here
                    
                    if 'sony_creative_style' in sony_tags:
                        tag_result['sony_has_creative_style'] = True
                        # Could add creative style analysis here
                    
                    # Group tags by category for easier analysis
                    lens_tags = {k: v for k, v in sony_tags.items() if 'lens' in k.lower()}
                    if lens_tags:
                        tag_result['sony_lens_tag_count'] = len(lens_tags)
                    
                    focus_tags = {k: v for k, v in sony_tags.items() if 'focus' in k.lower()}
                    if focus_tags:
                        tag_result['sony_focus_tag_count'] = len(focus_tags)
                return tag_result
            
            def extract_model_features():
                model_result = {}
                # Add Sony camera model specific features
                camera_model = exif_data.get('camera_model', '')
                
                # Sony Alpha series detection
                if 'ILCE' in camera_model or 'A7' in camera_model or 'A9' in camera_model:
                    model_result['sony_camera_series'] = 'Alpha'
                    
                    # Detect specific Alpha models
                    if 'A7R' in camera_model:
                        model_result['sony_high_resolution'] = True
                    elif 'A7S' in camera_model:
                        model_result['sony_high_sensitivity'] = True
                    elif 'A9' in camera_model:
                        model_result['sony_high_speed'] = True
                        
                # Sony RX series detection
                elif 'RX' in camera_model:
                    model_result['sony_camera_series'] = 'RX'
                    
                    # Detect specific RX models
                    if 'RX1' in camera_model:
                        model_result['sony_full_frame_compact'] = True
                    elif 'RX100' in camera_model:
                        model_result['sony_premium_compact'] = True
                    elif 'RX10' in camera_model:
                        model_result['sony_bridge_camera'] = True
                return model_result
            
            def extract_file_metadata():
                file_result = {}
                # Add file type
                file_result['file_type'] = 'ARW'
                
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
            
            # Count Sony-specific fields
            sony_fields = [key for key in result.keys() if key.startswith('sony_')]
            result['sony_field_count'] = len(sony_fields)
            
            # Log performance metrics
            end_time = time.time()
            memory_used, peak_memory, peak_percentage = tracker.end()
            logger.info(f"Extracted {len(sony_fields)} Sony-specific fields in {end_time - start_time:.2f} seconds")
            logger.info(f"Memory usage: {memory_used/(1024*1024):.2f} MB, Peak: {peak_memory/(1024*1024):.2f} MB ({peak_percentage:.1f}%)")
            
            return result
    
    @performance_monitor
    def process_raw(self, image_path: str, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Sony ARW file data
        
        Args:
            image_path: Path to the ARW file
            exif_data: Basic EXIF data already extracted
            
        Returns:
            Dictionary containing processed Sony ARW data
        """
        with MemoryTracker(self.memory_limit) as tracker:
            start_time = time.time()
            result = {}
            
            # Define parallel tasks for raw processing
            def extract_basic_metadata(raw):
                basic_result = {}
                try:
                    # Sony ARW version detection
                    basic_result['sony_arw_version'] = 'ARW 2.0' if str(raw.raw_type) == 'RawType.Flat' else 'ARW 1.0'
                    
                    # Try to get color profile information
                    if hasattr(raw, 'color_desc'):
                        basic_result['color_profile'] = raw.color_desc.decode('utf-8', errors='ignore')
                    
                    # Get white balance coefficients if available
                    if hasattr(raw, 'camera_whitebalance'):
                        # Check if it's a NumPy array (has tolist method) or already a list
                        if hasattr(raw.camera_whitebalance, 'tolist'):
                            basic_result['camera_white_balance'] = raw.camera_whitebalance.tolist()
                        else:
                            # If it's already a list or another type, use it directly
                            basic_result['camera_white_balance'] = raw.camera_whitebalance
                    
                    # Get raw image size
                    if hasattr(raw, 'sizes'):
                        basic_result['sony_raw_width'] = raw.sizes.raw_width
                        basic_result['sony_raw_height'] = raw.sizes.raw_height
                        
                    # Get black and white levels
                    basic_result['sony_raw_black_level'] = int(raw.black_level) if hasattr(raw, 'black_level') else 0
                    basic_result['sony_raw_white_level'] = int(raw.white_level) if hasattr(raw, 'white_level') else 0
                except Exception as e:
                    logger.error(f"Error extracting basic raw metadata: {e}")
                return basic_result
            
            def extract_raw_stats(raw):
                stats_result = {}
                try:
                    # Get raw image data - use sampling to reduce memory usage
                    raw_image = raw.raw_image
                    
                    # Ensure we're working with a numpy array
                    if not isinstance(raw_image, np.ndarray):
                        logger.warning(f"Raw image is not a numpy array, converting from {type(raw_image).__name__}")
                        # Try to convert to numpy array if it's a list
                        if isinstance(raw_image, list):
                            raw_image = np.array(raw_image)
                        else:
                            # If we can't convert, just return basic info
                            stats_result['sony_raw_image_type'] = str(type(raw_image).__name__)
                            stats_result['sony_raw_conversion_error'] = "Unable to convert to numpy array"
                            return stats_result
                    
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
                    stats_result['sony_raw_image_shape'] = str(raw_image.shape)
                    stats_result['sony_raw_min_value'] = int(raw_min)
                    stats_result['sony_raw_max_value'] = int(raw_max)
                    
                    # Calculate histogram with reduced bins for efficiency
                    histogram, _ = np.histogram(sampled_image.flatten(), bins=256)
                    
                    # Calculate basic statistics
                    stats_result['sony_raw_histogram_mean'] = float(np.mean(histogram))
                    stats_result['sony_raw_histogram_std'] = float(np.std(histogram))
                    stats_result['sony_raw_dynamic_range'] = float(np.log2(raw_max - raw_min + 1)) if raw_max > raw_min else 0
                    
                    # If GPU is available, do more advanced processing
                    if self.use_gpu:
                        try:
                            import torch
                            if torch.backends.mps.is_available():
                                # Convert to tensor and move to GPU
                                device = torch.device("mps")
                                # Ensure we're passing a numpy array to torch.tensor
                                tensor = torch.tensor(np.asarray(sampled_image), device=device)
                                
                                # Calculate percentiles for exposure analysis
                                percentiles = [1, 5, 10, 50, 90, 95, 99]
                                for p in percentiles:
                                    stats_result[f'sony_raw_percentile_{p}'] = float(torch.quantile(tensor.float(), p/100).cpu().numpy())
                                
                                # Calculate exposure metrics
                                highlight_threshold = 0.95 * raw_max
                                shadow_threshold = raw_min + 0.05 * (raw_max - raw_min)
                                
                                highlight_percentage = (tensor > highlight_threshold).sum().item() / tensor.numel() * 100
                                shadow_percentage = (tensor < shadow_threshold).sum().item() / tensor.numel() * 100
                                
                                stats_result['sony_highlight_percentage'] = round(highlight_percentage, 2)
                                stats_result['sony_shadow_percentage'] = round(shadow_percentage, 2)
                                stats_result['sony_midtone_percentage'] = round(100 - highlight_percentage - shadow_percentage, 2)
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
            
            # Add basic file info to result regardless of processing success
            result['file_path'] = image_path
            result['file_size'] = os.path.getsize(image_path) if os.path.exists(image_path) else 0
            result['sony_model'] = exif_data.get('camera_model', 'Unknown')
            
            # First try to use exiftool as a safer alternative
            try:
                import subprocess
                # Check if exiftool is available
                try:
                    exiftool_version = subprocess.run(['exiftool', '-ver'], 
                                                    capture_output=True, 
                                                    check=True, 
                                                    text=True).stdout.strip()
                    
                    # If exiftool is available, use it to extract basic metadata
                    exiftool_cmd = ['exiftool', '-json', '-g', image_path]
                    exiftool_output = subprocess.run(exiftool_cmd, 
                                                    capture_output=True, 
                                                    check=True, 
                                                    text=True).stdout
                    
                    import json
                    try:
                        exiftool_data = json.loads(exiftool_output)[0]
                        # Add basic exiftool data to result
                        result['exiftool_used'] = True
                        
                        # Extract key metadata from exiftool output
                        if 'File' in exiftool_data:
                            file_info = {f"file_{k.lower().replace(' ', '_')}": v 
                                        for k, v in exiftool_data['File'].items()}
                            result.update(file_info)
                            
                        if 'Sony' in exiftool_data:
                            sony_info = {f"sony_{k.lower().replace(' ', '_')}": v 
                                        for k, v in exiftool_data['Sony'].items()}
                            result.update(sony_info)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse exiftool JSON output")
                except (subprocess.SubprocessError, FileNotFoundError):
                    logger.warning("Exiftool not available")
            except Exception as e:
                logger.warning(f"Error using exiftool: {e}")
            
            # Now try to use rawpy with robust error handling
            try:
                # Try to open the raw file with a timeout to prevent hanging
                import signal
                
                # Define a timeout handler
                def timeout_handler(signum, frame):
                    raise TimeoutError("Timed out opening RAW file")
                
                # Set a timeout of 5 seconds for opening the file
                original_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(5)
                
                try:
                    with rawpy.imread(image_path) as raw:
                        # Cancel the alarm once file is opened
                        signal.alarm(0)
                        logger.info(f"Processing Sony ARW file: {image_path}")
                        
                        # Process the raw file in a safer way - one task at a time
                        # First extract basic metadata
                        basic_result = extract_basic_metadata(raw)
                        if basic_result:
                            result.update(basic_result)
                        
                        # Then try to extract thumbnail
                        thumb_result = extract_thumbnail(raw)
                        if thumb_result:
                            result.update(thumb_result)
                        
                        # Finally try to extract raw stats if previous steps succeeded
                        if not any(k.endswith('_error') for k in basic_result.keys()):
                            stats_result = extract_raw_stats(raw)
                            if stats_result:
                                result.update(stats_result)
                except TimeoutError as e:
                    logger.error(f"Timeout opening RAW file: {e}")
                    result['rawpy_timeout_error'] = str(e)
                finally:
                    # Reset the alarm handler
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, original_handler)
            except (rawpy.LibRawError, IOError, ValueError) as e:
                logger.error(f"Error opening RAW file with rawpy: {e}")
                result['rawpy_error'] = str(e)
            except Exception as e:
                logger.error(f"Sony ARW processing error: {e}")
                result['processing_error'] = str(e)
            
            # Log performance metrics
            end_time = time.time()
            memory_used, peak_memory, peak_percentage = tracker.end()
            logger.info(f"Processed Sony ARW data with {len(result)} attributes in {end_time - start_time:.2f} seconds")
            logger.info(f"Memory usage: {memory_used/(1024*1024):.2f} MB, Peak: {peak_memory/(1024*1024):.2f} MB ({peak_percentage:.1f}%)")
            
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
    
    @performance_monitor
    def process_thumbnail(self, thumb_data: bytes, thumb_format: str) -> Optional[Dict[str, Any]]:
        """Process Sony thumbnail data with GPU acceleration if available
        
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
