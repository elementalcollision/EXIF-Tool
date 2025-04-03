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
        
        # Check if exiftool is available
        self._exiftool_available = self._check_exiftool()
    
    def _check_exiftool(self):
        """Check if exiftool is available on the system
        
        Returns:
            bool: True if exiftool is available, False otherwise
        """
        try:
            result = subprocess.run(['which', 'exiftool'], 
                                  capture_output=True, 
                                  check=False)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Error checking for exiftool: {e}")
            return False
            
    def _run_exiftool(self, image_path):
        """Run exiftool on the image and return the extracted metadata
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: Dictionary containing the extracted metadata
        """
        try:
            if not hasattr(self, '_exiftool_available') or not self._exiftool_available:
                logger.warning("Exiftool not available")
                return {}
                
            # Run exiftool with JSON output
            exiftool_cmd = ['exiftool', '-j', '-a', '-u', '-G1', image_path]
            result = subprocess.run(exiftool_cmd, 
                                  capture_output=True, 
                                  check=False, 
                                  text=True)
            
            if result.returncode != 0:
                logger.error(f"Exiftool error: {result.stderr}")
                return {}
                
            # Parse JSON output
            try:
                import json
                data = json.loads(result.stdout)
                if data and isinstance(data, list) and len(data) > 0:
                    return data[0]
                return {}
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing exiftool output: {e}")
                return {}
        except Exception as e:
            logger.error(f"Error running exiftool: {e}")
            return {}
    
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
        # and set the camera make to FUJIFILM
        if 'camera_make' not in exif_data or not exif_data.get('camera_make'):
            # Set default camera make for RAF files
            exif_data['camera_make'] = 'FUJIFILM'
            is_fujifilm = True
            
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
                
                # Always try to use exiftool to get the exact camera model first
                try:
                    # First try to use exiftool to get the model
                    if self._check_exiftool():
                        logger.info(f"Using exiftool to extract exact camera model information")
                        # Use more detailed exiftool command to get all possible model information
                        exiftool_cmd = ['exiftool', '-Make', '-Model', '-CameraModelName', '-j', image_path]
                        exiftool_output = subprocess.run(exiftool_cmd, 
                                                       capture_output=True, 
                                                       check=False, 
                                                       text=True).stdout
                        try:
                            import json
                            exiftool_data = json.loads(exiftool_output)[0]
                            
                            # Try different fields that might contain the model information
                            if 'CameraModelName' in exiftool_data:
                                camera_model = exiftool_data['CameraModelName']
                            elif 'Model' in exiftool_data:
                                camera_model = exiftool_data['Model']
                                
                            if camera_model:
                                logger.info(f"Found camera model from exiftool: {camera_model}")
                                exif_data['camera_model'] = camera_model
                                model_result['camera_model'] = camera_model
                                
                            # Get camera make
                            if 'Make' in exiftool_data:
                                camera_make = exiftool_data['Make']
                                exif_data['camera_make'] = camera_make
                                model_result['camera_make'] = camera_make
                        except (json.JSONDecodeError, IndexError) as e:
                            logger.error(f"Error parsing exiftool output: {e}")
                except Exception as e:
                    logger.error(f"Error extracting camera model with exiftool: {e}")
                
                # If we still don't have a model, try to extract it from the filename
                if not camera_model:
                    filename = os.path.basename(image_path)
                    # Common Fujifilm model patterns in filenames
                    fuji_models = ['X-T1', 'X-T2', 'X-T3', 'X-T4', 'X-T5', 
                                   'X-Pro1', 'X-Pro2', 'X-Pro3', 
                                   'X-E1', 'X-E2', 'X-E3', 'X-E4',
                                   'X-H1', 'X-H2', 'X-H2S',
                                   'X100', 'X100S', 'X100T', 'X100F', 'X100V',
                                   'GFX50S', 'GFX50R', 'GFX100', 'GFX100S']
                    
                    for model in fuji_models:
                        if model.lower() in filename.lower():
                            camera_model = model
                            exif_data['camera_model'] = camera_model
                            model_result['camera_model'] = camera_model
                            break
                
                # If we still don't have a model, use a generic one based on the filename
                if not camera_model:
                    # Try to extract model from DSCF pattern (common in Fujifilm filenames)
                    if filename.startswith('DSCF'):
                        camera_model = 'Fujifilm X Series'
                        exif_data['camera_model'] = camera_model
                        model_result['camera_model'] = camera_model
                
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
            
            # Add basic file info to result regardless of processing success
            result['file_path'] = image_path
            result['file_size'] = os.path.getsize(image_path) if os.path.exists(image_path) else 0
            
            # Ensure camera make and model are set correctly
            camera_make = exif_data.get('camera_make', '')
            camera_model = exif_data.get('camera_model', '')
            
            # Always try to use exiftool to get the exact camera model first
            try:
                # Use direct exiftool command without relying on helper methods
                import subprocess
                import json
                
                logger.info(f"Using exiftool to extract exact camera model information")
                # Use more detailed exiftool command to get all possible model information
                exiftool_cmd = ['exiftool', '-Make', '-Model', '-CameraModelName', '-j', image_path]
                exiftool_process = subprocess.run(
                    exiftool_cmd,
                    capture_output=True,
                    check=False,
                    text=True
                )
                
                if exiftool_process.returncode == 0 and exiftool_process.stdout:
                    try:
                        exiftool_data = json.loads(exiftool_process.stdout)[0]
                        
                        # Try different fields that might contain the model information
                        if 'CameraModelName' in exiftool_data:
                            camera_model = exiftool_data['CameraModelName']
                            logger.info(f"Found camera model from CameraModelName: {camera_model}")
                        elif 'Model' in exiftool_data:
                            camera_model = exiftool_data['Model']
                            logger.info(f"Found camera model from Model: {camera_model}")
                            
                        if camera_model:
                            exif_data['camera_model'] = camera_model
                        
                        # Get camera make
                        if 'Make' in exiftool_data:
                            camera_make = exiftool_data['Make']
                            exif_data['camera_make'] = camera_make
                            logger.info(f"Found camera make: {camera_make}")
                    except (json.JSONDecodeError, IndexError) as e:
                        logger.error(f"Error parsing exiftool output: {e}")
                else:
                    logger.error(f"Exiftool failed: {exiftool_process.stderr}")
            except Exception as e:
                logger.error(f"Error extracting camera model with exiftool: {e}")
            
            # If camera make is still not available or empty, set it to FUJIFILM
            if not camera_make:
                camera_make = 'FUJIFILM'
                exif_data['camera_make'] = camera_make
            
            # If camera model is still not available or empty, try to determine it from the filename
            if not camera_model:
                filename = os.path.basename(image_path)
                # Common Fujifilm model patterns in filenames
                fuji_models = ['X-T1', 'X-T2', 'X-T3', 'X-T4', 'X-T5', 
                               'X-Pro1', 'X-Pro2', 'X-Pro3', 
                               'X-E1', 'X-E2', 'X-E3', 'X-E4',
                               'X-H1', 'X-H2', 'X-H2S',
                               'X100', 'X100S', 'X100T', 'X100F', 'X100V',
                               'GFX50S', 'GFX50R', 'GFX100', 'GFX100S', 'GFX100S II']
                
                for model in fuji_models:
                    if model.lower() in filename.lower():
                        camera_model = model
                        exif_data['camera_model'] = camera_model
                        break
                
                # If we still don't have a model, use a generic one based on the filename
                if not camera_model and filename.startswith('DSCF'):
                    camera_model = 'Fujifilm X Series'
                    exif_data['camera_model'] = camera_model
            
            # Set camera make and model in the result
            result['camera_make'] = camera_make
            result['camera_model'] = camera_model
            result['fuji_model'] = camera_model
            
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
                            
                        if 'FujiFilm' in exiftool_data:
                            fuji_info = {f"fuji_{k.lower().replace(' ', '_')}": v 
                                        for k, v in exiftool_data['FujiFilm'].items()}
                            result.update(fuji_info)
                            
                        # Add raw processing status
                        result['raw_processing_status'] = 'exiftool_primary'
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse exiftool JSON output")
                except (subprocess.SubprocessError, FileNotFoundError):
                    logger.warning("Exiftool not available")
            except Exception as e:
                logger.warning(f"Error using exiftool: {e}")
            
            # Process RAF file with rawpy with robust error handling
            try:
                import rawpy
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
                        logger.info(f"Processing Fujifilm RAF file: {image_path}")
                        
                        # Process the raw file in a safer way - one task at a time
                        # First extract basic metadata
                        basic_result = extract_basic_metadata(raw)
                        if basic_result:
                            result.update(basic_result)
                            result['raw_processing_status'] = 'rawpy_success'
                        
                        # Then try to extract thumbnail
                        thumb_result = extract_thumbnail(raw)
                        if thumb_result:
                            result.update(thumb_result)
                        
                        # Finally try to extract raw stats if previous steps succeeded
                        if not any(k.endswith('_error') for k in result.keys()):
                            stats_result = extract_raw_stats(raw)
                            if stats_result:
                                result.update(stats_result)
                except TimeoutError as e:
                    logger.error(f"Timeout opening RAW file: {e}")
                    result['rawpy_timeout_error'] = str(e)
                    result['raw_processing_status'] = 'timeout'
                except (rawpy.LibRawError, ValueError, IOError) as e:
                    logger.error(f"Error opening RAW file with rawpy: {e}")
                    result['rawpy_error'] = str(e)
                    result['raw_processing_status'] = 'failed'
                finally:
                    # Reset the alarm handler
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, original_handler)
            except ImportError:
                logger.error("rawpy not available for RAF processing")
                result['rawpy_import_error'] = "rawpy module not available"
            except Exception as e:
                logger.error(f"Fujifilm RAF processing error: {e}")
                result['processing_error'] = str(e)
            
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
