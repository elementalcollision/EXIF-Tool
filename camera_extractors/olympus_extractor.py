#!/usr/bin/env python3
"""
Olympus Camera Extractor
Provides Olympus-specific EXIF extraction for ORF files
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
    MemoryTracker, performance_monitor, safe_array_operation, ThreadPoolManager
)

# Configure logging
logger = logging.getLogger('olympus_extractor')


class OlympusExtractor(CameraExtractor):
    """Olympus-specific EXIF extractor for ORF files"""
    
    def __init__(self, use_gpu=False, memory_limit=0.75, cpu_cores=None):
        super().__init__(use_gpu=use_gpu, memory_limit=memory_limit, cpu_cores=cpu_cores)
        
        # Initialize thread pool
        self.thread_pool = ThreadPoolManager(cpu_cores)
        
        # Log initialization
        logger.info(f"Initializing OlympusExtractor with GPU={use_gpu}, "
                  f"memory_limit={memory_limit}, cpu_cores={cpu_cores}")
                  
    def _check_exiftool(self):
        """Check if exiftool is available"""
        try:
            result = subprocess.run(['exiftool', '-ver'], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE, 
                                  text=True, 
                                  check=False)
            return result.returncode == 0
        except Exception:
            return False
            
    def _run_exiftool(self, image_path):
        """Run exiftool and get results as JSON"""
        try:
            result = subprocess.run(
                ['exiftool', '-j', '-n', image_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                if data and isinstance(data, list) and len(data) > 0:
                    return data[0]
        except Exception as e:
            logger.error(f"Error running exiftool: {e}")
        
        return None
        
    def _extract_preview(self, image_path):
        """Extract preview image from RAW file using exiftool"""
        try:
            # Create a temporary file to store the preview
            fd, preview_path = tempfile.mkstemp(suffix='.jpg')
            os.close(fd)
            
            # Run exiftool to extract preview
            result = subprocess.run(
                ['exiftool', '-b', '-PreviewImage', '-w', preview_path, image_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode == 0 and os.path.exists(preview_path) and os.path.getsize(preview_path) > 0:
                return preview_path
            else:
                if os.path.exists(preview_path):
                    os.remove(preview_path)
                return None
        except Exception as e:
            logger.error(f"Error extracting preview: {e}")
            return None
    
    def can_handle(self, file_ext: str, exif_data: Dict[str, Any]) -> bool:
        """Check if this extractor can handle the given file
        
        Args:
            file_ext: File extension (e.g., '.orf')
            exif_data: Basic EXIF data already extracted
            
        Returns:
            True if this is an Olympus/OM System ORF file, False otherwise
        """
        # Check file extension
        if file_ext.lower() != '.orf':
            return False
            
        # Check camera make if available
        camera_make = exif_data.get('camera_make', '').upper()
        camera_model = exif_data.get('camera_model', '').upper()
        
        # Check for Olympus or OM Digital Solutions brands
        is_olympus = any(brand in camera_make for brand in ['OLYMPUS', 'OM DIGITAL', 'OM SYSTEM', 'OM-'])
        
        # Also check model for OM System cameras (OM-1, OM-5, etc.)
        is_om_model = any(model in camera_model for brand in ['OM-1', 'OM-5', 'OM-', 'E-M'] 
                          for model in [f"{brand}", f"{brand} "])
        
        logger.info(f"Checking if camera make '{exif_data.get('camera_make', '')}' "
                  f"model '{exif_data.get('camera_model', '')}' is Olympus/OM System: {is_olympus or is_om_model}")
        
        # If camera_make is not available but it's an ORF file, assume it's Olympus
        if 'camera_make' not in exif_data:
            logger.info(f"No camera make found, assuming ORF file is from Olympus/OM System")
            return True
            
        return is_olympus or is_om_model
    
    @performance_monitor
    def extract_metadata(self, image_path: str, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Olympus-specific metadata from the image
        
        Args:
            image_path: Path to the image file
            exif_data: Basic EXIF data already extracted
            
        Returns:
            Dictionary containing Olympus-specific metadata
        """
        with MemoryTracker(self.memory_limit) as tracker:
            start_time = time.time()
            result = {}
            
            # Define tasks to run in parallel
            def extract_exiftool_data():
                exiftool_result = {}
                try:
                    if self._check_exiftool():
                        logger.info("Using exiftool to extract Olympus metadata")
                        exiftool_data = self._run_exiftool(image_path)
                        if exiftool_data:
                            # Process Olympus-specific tags
                            for key, value in exiftool_data.items():
                                if key.startswith('Olympus'):
                                    # Convert to snake_case
                                    olympus_key = 'olympus_' + key[7:].lower().replace(' ', '_')
                                    exiftool_result[olympus_key] = value
                            
                            # Extract important Olympus metadata
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
                            
                            # Extract Olympus-specific features
                            olympus_specific = {
                                'OlympusCameraType': 'olympus_camera_type',
                                'OlympusImageHeight': 'olympus_image_height',
                                'OlympusImageWidth': 'olympus_image_width',
                                'OlympusEquipment': 'olympus_equipment',
                                'OlympusCameraSettings': 'olympus_camera_settings',
                                'OlympusRawDevelopment': 'olympus_raw_development',
                                'OlympusFocusInfo': 'olympus_focus_info',
                                'OlympusImageProcessing': 'olympus_image_processing'
                            }
                            
                            for exif_key, result_key in olympus_specific.items():
                                if exif_key in exiftool_data:
                                    exiftool_result[result_key] = exiftool_data[exif_key]
                except Exception as e:
                    logger.error(f"Error extracting Olympus metadata with exiftool: {e}")
                return exiftool_result
            
            def extract_model_features():
                model_result = {}
                # Add Olympus/OM System camera model specific features
                camera_make = exif_data.get('camera_make', '')
                camera_model = exif_data.get('camera_model', '')
                
                # Identify manufacturer
                if 'OM DIGITAL' in camera_make.upper() or 'OM SYSTEM' in camera_make.upper():
                    model_result['olympus_manufacturer'] = 'OM Digital Solutions'
                    model_result['olympus_om_system'] = True
                else:
                    model_result['olympus_manufacturer'] = 'Olympus'
                
                # Detect camera series
                if 'OM-1' in camera_model or 'OM-5' in camera_model:
                    model_result['olympus_camera_series'] = 'OM System'
                    model_result['olympus_mirrorless'] = True
                    model_result['olympus_generation'] = 'Current'
                    model_result['olympus_sensor_type'] = 'Stacked CMOS'
                    
                    # OM-1 specific features
                    if 'OM-1' in camera_model:
                        model_result['olympus_computational_features'] = True
                        model_result['olympus_high_res_mode'] = True
                        model_result['olympus_weather_sealing'] = 'IP53'
                elif 'OM-D' in camera_model:
                    model_result['olympus_camera_series'] = 'OM-D'
                    model_result['olympus_mirrorless'] = True
                    
                    # Identify specific OM-D models
                    if 'E-M1' in camera_model:
                        model_result['olympus_tier'] = 'Professional'
                        if 'MARK III' in camera_model.upper() or 'MK3' in camera_model.upper():
                            model_result['olympus_generation'] = 'Latest'
                    elif 'E-M5' in camera_model:
                        model_result['olympus_tier'] = 'Enthusiast'
                    elif 'E-M10' in camera_model:
                        model_result['olympus_tier'] = 'Consumer'
                elif 'PEN' in camera_model:
                    model_result['olympus_camera_series'] = 'PEN'
                    model_result['olympus_mirrorless'] = True
                    model_result['olympus_form_factor'] = 'Compact'
                elif 'E-' in camera_model:
                    model_result['olympus_camera_series'] = 'E-System'
                    model_result['olympus_dslr'] = True
                elif 'TOUGH' in camera_model:
                    model_result['olympus_camera_series'] = 'TOUGH'
                    model_result['olympus_compact'] = True
                    model_result['olympus_rugged'] = True
                
                # Add sensor information if available
                if 'OM-1' in camera_model:
                    model_result['olympus_sensor_mp'] = 20.4
                    model_result['olympus_sensor_size'] = 'Micro Four Thirds'
                    model_result['olympus_sensor_dimensions'] = '17.4 x 13.0 mm'
                elif 'E-M1' in camera_model and ('MARK III' in camera_model or 'MK3' in camera_model):
                    model_result['olympus_sensor_mp'] = 20.4
                    model_result['olympus_sensor_size'] = 'Micro Four Thirds'
                    model_result['olympus_sensor_dimensions'] = '17.4 x 13.0 mm'
                
                return model_result
            
            def extract_file_metadata():
                file_result = {}
                # Add file type
                file_result['file_type'] = 'ORF'
                
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
            
            # Count Olympus-specific fields
            olympus_fields = [key for key in result.keys() if key.startswith('olympus_')]
            result['olympus_field_count'] = len(olympus_fields)
            
            # Log performance metrics
            end_time = time.time()
            memory_used, peak_memory, peak_percentage = tracker.end()
            logger.info(f"Extracted {len(olympus_fields)} Olympus-specific fields in {end_time - start_time:.2f} seconds")
            logger.info(f"Memory usage: {memory_used/(1024*1024):.2f} MB, Peak: {peak_memory/(1024*1024):.2f} MB ({peak_percentage:.1f}%)")
            
            return result
    
    @performance_monitor
    def process_raw(self, image_path: str, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Olympus RAW file data
        
        Args:
            image_path: Path to the ORF file
            exif_data: Basic EXIF data already extracted
            
        Returns:
            Dictionary containing processed Olympus RAW data
        """
        with MemoryTracker(self.memory_limit) as tracker:
            start_time = time.time()
            result = {}
            
            logger.info(f"Processing Olympus ORF specific data")
            
            # Define parallel tasks for ORF processing
            def extract_basic_metadata(raw):
                basic_result = {}
                try:
                    # Get basic metadata
                    if hasattr(raw, 'color_desc'):
                        basic_result['olympus_color_profile'] = raw.color_desc.decode('utf-8', errors='ignore')
                    
                    # Get white balance coefficients if available
                    if hasattr(raw, 'camera_whitebalance'):
                        if hasattr(raw.camera_whitebalance, 'tolist'):
                            basic_result['olympus_camera_whitebalance'] = raw.camera_whitebalance.tolist()
                        else:
                            basic_result['olympus_camera_whitebalance'] = str(raw.camera_whitebalance)
                    
                    # Get image dimensions
                    if hasattr(raw, 'sizes'):
                        basic_result['olympus_full_width'] = raw.sizes.width
                        basic_result['olympus_full_height'] = raw.sizes.height
                        basic_result['olympus_raw_width'] = raw.sizes.raw_width
                        basic_result['olympus_raw_height'] = raw.sizes.raw_height
                    
                    # Get black and white levels
                    basic_result['olympus_raw_black_level'] = int(raw.black_level) if hasattr(raw, 'black_level') else 0
                    basic_result['olympus_raw_white_level'] = int(raw.white_level) if hasattr(raw, 'white_level') else 0
                except Exception as e:
                    logger.error(f"Error extracting basic raw metadata: {e}")
                return basic_result
            
            def extract_raw_stats(raw):
                stats_result = {}
                try:
                    # Define a function to process safely
                    def process_stats():
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
                        stats_result['olympus_raw_image_shape'] = str(raw_image.shape)
                        stats_result['olympus_raw_min_value'] = int(raw_min)
                        stats_result['olympus_raw_max_value'] = int(raw_max)
                        
                        # Calculate histogram with reduced bins for efficiency
                        histogram, _ = np.histogram(sampled_image.flatten(), bins=256)
                        
                        # Calculate basic statistics
                        stats_result['olympus_raw_histogram_mean'] = float(np.mean(histogram))
                        stats_result['olympus_raw_histogram_std'] = float(np.std(histogram))
                        stats_result['olympus_raw_dynamic_range'] = float(np.log2(raw_max - raw_min + 1)) if raw_max > raw_min else 0
                    
                    # Execute the stats processing function safely
                    safe_array_operation(process_stats)
                    
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
                                    stats_result[f'olympus_raw_percentile_{p}'] = float(torch.quantile(tensor.float(), p/100).cpu().numpy())
                                
                                # Calculate exposure metrics
                                highlight_threshold = 0.95 * raw_max
                                shadow_threshold = raw_min + 0.05 * (raw_max - raw_min)
                                
                                highlight_percentage = (tensor > highlight_threshold).sum().item() / tensor.numel() * 100
                                shadow_percentage = (tensor < shadow_threshold).sum().item() / tensor.numel() * 100
                                
                                stats_result['olympus_highlight_percentage'] = round(highlight_percentage, 2)
                                stats_result['olympus_shadow_percentage'] = round(shadow_percentage, 2)
                                stats_result['olympus_midtone_percentage'] = round(100 - highlight_percentage - shadow_percentage, 2)
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
            result['olympus_model'] = exif_data.get('camera_model', 'Unknown')
            
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
                            
                        if 'Olympus' in exiftool_data:
                            olympus_info = {f"olympus_{k.lower().replace(' ', '_')}": v 
                                        for k, v in exiftool_data['Olympus'].items()}
                            result.update(olympus_info)
                            
                        # Extract image dimensions if available
                        if 'ImageWidth' in exiftool_data and 'ImageHeight' in exiftool_data:
                            result['olympus_image_width'] = exiftool_data['ImageWidth']
                            result['olympus_image_height'] = exiftool_data['ImageHeight']
                            result['olympus_megapixels'] = round(exiftool_data['ImageWidth'] * exiftool_data['ImageHeight'] / 1000000, 1)
                        
                        # Extract color space information
                        if 'ColorSpace' in exiftool_data:
                            result['olympus_color_space'] = exiftool_data['ColorSpace']
                        
                        # Extract bit depth if available
                        if 'BitsPerSample' in exiftool_data:
                            result['olympus_bits_per_sample'] = exiftool_data['BitsPerSample']
                        
                        # Extract compression information
                        if 'Compression' in exiftool_data:
                            result['olympus_compression'] = exiftool_data['Compression']
                        
                        # Get ORF version if available
                        if 'ORFVersion' in exiftool_data:
                            result['olympus_orf_version'] = exiftool_data['ORFVersion']
                        elif 'RAWVersion' in exiftool_data:
                            result['olympus_raw_version'] = exiftool_data['RAWVersion']
                            
                        # Add raw processing status
                        result['raw_processing_status'] = 'exiftool_primary'
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse exiftool JSON output")
                except (subprocess.SubprocessError, FileNotFoundError):
                    logger.warning("Exiftool not available")
            except Exception as e:
                logger.warning(f"Error using exiftool: {e}")
            
            # Process ORF file with rawpy with robust error handling
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
                        logger.info(f"Processing Olympus ORF file: {image_path}")
                        
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
                    logger.error(f"Error opening ORF file with rawpy: {e}")
                    result['rawpy_error'] = str(e)
                    result['raw_processing_status'] = 'failed'
                    
                    # Try to extract preview image using exiftool as fallback
                    logger.info("Falling back to exiftool for preview extraction")
                    preview_path = self._extract_preview(image_path)
                    if preview_path:
                            # Extract image dimensions if available
                            if 'ImageWidth' in exiftool_data and 'ImageHeight' in exiftool_data:
                                result['olympus_image_width'] = exiftool_data['ImageWidth']
                                result['olympus_image_height'] = exiftool_data['ImageHeight']
                                result['olympus_megapixels'] = round(exiftool_data['ImageWidth'] * exiftool_data['ImageHeight'] / 1000000, 1)
                            
                            # Extract color space information
                            if 'ColorSpace' in exiftool_data:
                                result['olympus_color_space'] = exiftool_data['ColorSpace']
                            
                            # Extract bit depth if available
                            if 'BitsPerSample' in exiftool_data:
                                result['olympus_bits_per_sample'] = exiftool_data['BitsPerSample']
                            
                            # Extract compression information
                            if 'Compression' in exiftool_data:
                                result['olympus_compression'] = exiftool_data['Compression']
                            
                            # Get ORF version if available
                            if 'ORFVersion' in exiftool_data:
                                result['olympus_orf_version'] = exiftool_data['ORFVersion']
                            elif 'RAWVersion' in exiftool_data:
                                result['olympus_raw_version'] = exiftool_data['RAWVersion']
                                
                            # Extract OM Digital Solutions specific data
                            if 'Model' in exiftool_data and ('OM-1' in exiftool_data['Model'] or 'OM-5' in exiftool_data['Model']):
                                # Extract computational photography features
                                if 'ComputationalPhotography' in exiftool_data:
                                    result['olympus_computational_photography'] = exiftool_data['ComputationalPhotography']
                                
                                # Extract high-res mode information
                                if 'HighResMode' in exiftool_data:
                                    result['olympus_high_res_mode'] = exiftool_data['HighResMode']
                                
                                # Extract AI detection information
                                if 'AIDetection' in exiftool_data:
                                    result['olympus_ai_detection'] = exiftool_data['AIDetection']
                            
                            # Try to extract preview image
                            preview_path = self._extract_preview(image_path)
                            if preview_path:
                                try:
                                    with Image.open(preview_path) as preview:
                                        result['olympus_preview_width'] = preview.width
                                        result['olympus_preview_height'] = preview.height
                                        result['olympus_preview_format'] = preview.format
                                        logger.info(f"Extracted preview image: {preview.width}x{preview.height} {preview.format}")
                                except Exception as e:
                                    logger.error(f"Preview processing error: {e}")
                                finally:
                                    if os.path.exists(preview_path):
                                        os.remove(preview_path)
            except ImportError:
                logger.error("rawpy not available for ORF processing")
            except Exception as e:
                logger.error(f"Olympus ORF processing error: {e}")
            
            # Log performance metrics
            end_time = time.time()
            memory_used, peak_memory, peak_percentage = tracker.end()
            logger.info(f"Processed Olympus ORF data with {len(result)} attributes in {end_time - start_time:.2f} seconds")
            logger.info(f"Memory usage: {memory_used/(1024*1024):.2f} MB, Peak: {peak_memory/(1024*1024):.2f} MB ({peak_percentage:.1f}%)")
            
            return result
    
    def get_makernote_tags(self) -> Dict[str, str]:
        """Get Olympus MakerNote tag mapping
        
        Returns:
            Dictionary mapping Olympus MakerNote tags to human-readable names
        """
        return {
            "0x0100": "ThumbnailImage",
            "0x0200": "SpecialMode",
            "0x0201": "CompressionMode",
            "0x0202": "MacroMode",
            "0x0203": "BWMode",
            "0x0204": "DigitalZoom",
            "0x0205": "FocalPlaneDiagonal",
            "0x0206": "LensDistortionParams",
            "0x0207": "CameraType",
            "0x0208": "TextInfo",
            "0x0209": "CameraID",
            "0x0300": "PreCaptureFrames",
            "0x0404": "SerialNumber",
            "0x1000": "ShutterSpeedValue",
            "0x1001": "ISOValue",
            "0x1002": "ApertureValue",
            "0x1003": "BrightnessValue",
            "0x1004": "FlashMode",
            "0x1005": "FlashDevice",
            "0x1006": "ExposureCompensation",
            "0x1007": "SensorTemperature",
            "0x1008": "LensTemperature",
            "0x100b": "FocusMode",
            "0x100c": "ManualFocusDistance",
            "0x100d": "ZoomStepCount",
            "0x100e": "FocusStepCount",
            "0x100f": "SharpnessMode",
            "0x1010": "FlashChargeLevel",
            "0x1011": "ColorMatrix",
            "0x1012": "BlackLevel",
            "0x1015": "WhiteBalance",
            "0x1017": "RedBalance",
            "0x1018": "BlueBalance",
            "0x101a": "SerialNumber",
            "0x1023": "FlashBias",
            "0x1029": "ExternalFlashBounce",
            "0x102a": "ExternalFlashZoom",
            "0x102c": "ExternalFlashMode",
            "0x1039": "Contrast",
            "0x103a": "SharpnessFactor",
            "0x103b": "ColorControl",
            "0x103c": "ValidBits",
            "0x103d": "CoringFilter",
            "0x103e": "ImageWidth",
            "0x103f": "ImageHeight",
            "0x1040": "CompressionRatio"
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
