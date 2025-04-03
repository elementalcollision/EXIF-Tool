#!/usr/bin/env python3
"""
Canon Camera Extractor
Provides Canon-specific EXIF extraction for CR2 and CR3 files
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
logger = logging.getLogger('canon_extractor')


class CanonExtractor(CameraExtractor):
    """Canon-specific EXIF extractor for CR2 and CR3 files"""
    
    def __init__(self, use_gpu=False, memory_limit=0.75, cpu_cores=None):
        super().__init__(use_gpu=use_gpu, memory_limit=memory_limit, cpu_cores=cpu_cores)
        
        # Log initialization
        logger.info(f"Initializing CanonExtractor with GPU={use_gpu}, "
                  f"memory_limit={memory_limit}, cpu_cores={cpu_cores}")
    
    def can_handle(self, file_ext: str, exif_data: Dict[str, Any]) -> bool:
        """Check if this extractor can handle the given file
        
        Args:
            file_ext: File extension (e.g., '.cr2', '.cr3')
            exif_data: Basic EXIF data already extracted
            
        Returns:
            True if this is a Canon CR2/CR3 file, False otherwise
        """
        # For CR3 files, always return True since they're Canon-specific
        if file_ext.lower() == '.cr3':
            logger.info("Detected Canon CR3 file")
            return True
            
        # For other Canon RAW formats, check camera make if available
        is_canon = exif_data.get('camera_make', '').upper().startswith('CANON')
        is_cr_file = file_ext.lower() in ['.cr2', '.crw']
        
        # If camera_make is not available but it's a Canon RAW format, assume it's Canon
        if 'camera_make' not in exif_data and is_cr_file:
            return True
            
        return is_canon and is_cr_file
    
    @performance_monitor
    def extract_metadata(self, image_path: str, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Canon-specific metadata from the image
        
        Args:
            image_path: Path to the image file
            exif_data: Basic EXIF data already extracted
            
        Returns:
            Dictionary containing Canon-specific metadata
        """
        with MemoryTracker(self.memory_limit) as tracker:
            start_time = time.time()
            result = {}
            file_ext = os.path.splitext(image_path)[1].lower()
            
            # Define tasks to run in parallel
            def extract_exiftool_data():
                exiftool_result = {}
                try:
                    if self._check_exiftool():
                        logger.info("Using exiftool to extract Canon metadata")
                        exiftool_data = self._run_exiftool(image_path)
                        if exiftool_data:
                            # Process Canon-specific tags
                            for key, value in exiftool_data.items():
                                if key.startswith('Canon'):
                                    # Convert to snake_case
                                    canon_key = 'canon_' + key[5:].lower().replace(' ', '_')
                                    exiftool_result[canon_key] = value
                            
                            # Extract important Canon metadata
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
                            
                            # Extract CR3-specific metadata
                            if file_ext == '.cr3':
                                if 'ImageWidth' in exiftool_data:
                                    exiftool_result['width'] = exiftool_data['ImageWidth']
                                if 'ImageHeight' in exiftool_data:
                                    exiftool_result['height'] = exiftool_data['ImageHeight']
                                if 'CanonModelID' in exiftool_data:
                                    exiftool_result['canon_model_id'] = exiftool_data['CanonModelID']
                                if 'CanonFirmwareVersion' in exiftool_data:
                                    exiftool_result['canon_firmware'] = exiftool_data['CanonFirmwareVersion']
                except Exception as e:
                    logger.error(f"Error extracting Canon metadata with exiftool: {e}")
                return exiftool_result
            
            def extract_model_features():
                model_result = {}
                # Add Canon camera model specific features
                camera_model = exif_data.get('camera_model', '')
                
                # Detect Canon camera series
                if 'EOS' in camera_model:
                    model_result['canon_camera_series'] = 'EOS'
                    
                    # Detect specific EOS models
                    if any(x in camera_model for x in ['R5', 'R6', 'R3']):
                        model_result['canon_mirrorless'] = True
                    if any(x in camera_model for x in ['5D', '1D']):
                        model_result['canon_professional'] = True
                    if any(x in camera_model for x in ['90D', '80D', '70D']):
                        model_result['canon_enthusiast'] = True
                    if any(x in camera_model for x in ['Rebel', '2000D', '1500D']):
                        model_result['canon_consumer'] = True
                elif 'PowerShot' in camera_model:
                    model_result['canon_camera_series'] = 'PowerShot'
                    model_result['canon_compact'] = True
                return model_result
            
            def extract_file_metadata():
                file_result = {}
                # Add file type
                file_result['file_type'] = file_ext.upper()[1:]
                
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
            
            # Count Canon-specific fields
            canon_fields = [key for key in result.keys() if key.startswith('canon_')]
            result['canon_field_count'] = len(canon_fields)
            
            # Log performance metrics
            end_time = time.time()
            memory_used, peak_memory, peak_percentage = tracker.end()
            logger.info(f"Extracted {len(canon_fields)} Canon-specific fields in {end_time - start_time:.2f} seconds")
            logger.info(f"Memory usage: {memory_used/(1024*1024):.2f} MB, Peak: {peak_memory/(1024*1024):.2f} MB ({peak_percentage:.1f}%)")
            
            return result
    
    @performance_monitor
    def process_raw(self, image_path: str, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Canon RAW file data
        
        Args:
            image_path: Path to the CR2/CR3 file
            exif_data: Basic EXIF data already extracted
            
        Returns:
            Dictionary containing processed Canon RAW data
        """
        with MemoryTracker(self.memory_limit) as tracker:
            start_time = time.time()
            result = {}
            file_ext = os.path.splitext(image_path)[1].lower()
            
            logger.info(f"Processing Canon {file_ext} specific data")
            
            # CR3 files need special handling
            if file_ext == '.cr3':
                def process_cr3():
                    cr3_result = {}
                    # Try to extract preview image using exiftool
                    if self._check_exiftool():
                        preview_path = self._extract_preview(image_path)
                        if preview_path:
                            try:
                                with Image.open(preview_path) as preview:
                                    width, height = preview.size
                                    cr3_result['canon_preview_width'] = width
                                    cr3_result['canon_preview_height'] = height
                                    cr3_result['canon_preview_format'] = preview.format
                                    logger.info(f"Extracted preview image: {width}x{height} {preview.format}")
                                    
                                    # Process preview with GPU if available
                                    if self.use_gpu:
                                        preview_data = self._process_preview_with_gpu(preview)
                                        if preview_data:
                                            cr3_result.update(preview_data)
                            except Exception as e:
                                logger.error(f"Preview processing error: {e}")
                            finally:
                                # Clean up temporary preview file
                                if os.path.exists(preview_path):
                                    os.remove(preview_path)
                    return cr3_result
                
                # Execute CR3 processing
                try:
                    cr3_result = process_cr3()
                    result.update(cr3_result)
                except Exception as e:
                    logger.error(f"Error processing CR3 file: {e}")
            
            # CR2 files can be processed with rawpy
            elif file_ext == '.cr2':
                # Define parallel tasks for CR2 processing
                def extract_basic_metadata(raw):
                    basic_result = {}
                    try:
                        # Get basic metadata
                        if hasattr(raw, 'color_desc'):
                            basic_result['canon_color_profile'] = raw.color_desc.decode('utf-8', errors='ignore')
                        
                        # Get white balance coefficients if available
                        if hasattr(raw, 'camera_whitebalance'):
                            if hasattr(raw.camera_whitebalance, 'tolist'):
                                basic_result['canon_camera_whitebalance'] = raw.camera_whitebalance.tolist()
                            else:
                                basic_result['canon_camera_whitebalance'] = str(raw.camera_whitebalance)
                        
                        # Get image dimensions
                        if hasattr(raw, 'sizes'):
                            basic_result['canon_full_width'] = raw.sizes.width
                            basic_result['canon_full_height'] = raw.sizes.height
                            basic_result['canon_raw_width'] = raw.sizes.raw_width
                            basic_result['canon_raw_height'] = raw.sizes.raw_height
                        
                        # Get black and white levels
                        basic_result['canon_raw_black_level'] = int(raw.black_level) if hasattr(raw, 'black_level') else 0
                        basic_result['canon_raw_white_level'] = int(raw.white_level) if hasattr(raw, 'white_level') else 0
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
                            stats_result['canon_raw_image_shape'] = str(raw_image.shape)
                            stats_result['canon_raw_min_value'] = int(raw_min)
                            stats_result['canon_raw_max_value'] = int(raw_max)
                            
                            # Calculate histogram with reduced bins for efficiency
                            histogram, _ = np.histogram(sampled_image.flatten(), bins=256)
                            
                            # Calculate basic statistics
                            stats_result['canon_raw_histogram_mean'] = float(np.mean(histogram))
                            stats_result['canon_raw_histogram_std'] = float(np.std(histogram))
                            stats_result['canon_raw_dynamic_range'] = float(np.log2(raw_max - raw_min + 1)) if raw_max > raw_min else 0
                            
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
                                            stats_result[f'canon_raw_percentile_{p}'] = float(torch.quantile(tensor.float(), p/100).cpu().numpy())
                                        
                                        # Calculate exposure metrics
                                        highlight_threshold = 0.95 * raw_max
                                        shadow_threshold = raw_min + 0.05 * (raw_max - raw_min)
                                        
                                        highlight_percentage = (tensor > highlight_threshold).sum().item() / tensor.numel() * 100
                                        shadow_percentage = (tensor < shadow_threshold).sum().item() / tensor.numel() * 100
                                        
                                        stats_result['canon_highlight_percentage'] = round(highlight_percentage, 2)
                                        stats_result['canon_shadow_percentage'] = round(shadow_percentage, 2)
                                        stats_result['canon_midtone_percentage'] = round(100 - highlight_percentage - shadow_percentage, 2)
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
                result['canon_model'] = exif_data.get('camera_model', 'Unknown')
                
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
                                
                            if 'Canon' in exiftool_data:
                                canon_info = {f"canon_{k.lower().replace(' ', '_')}": v 
                                            for k, v in exiftool_data['Canon'].items()}
                                result.update(canon_info)
                                
                            # Add raw processing status
                            result['raw_processing_status'] = 'exiftool_primary'
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse exiftool JSON output")
                    except (subprocess.SubprocessError, FileNotFoundError):
                        logger.warning("Exiftool not available")
                except Exception as e:
                    logger.warning(f"Error using exiftool: {e}")
                
                # Process CR2 file with rawpy with robust error handling
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
                            logger.info(f"Processing Canon CR2 file: {image_path}")
                            
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
                    logger.error("rawpy not available for CR2 processing")
                    result['rawpy_import_error'] = "rawpy module not available"
                except Exception as e:
                    logger.error(f"Canon CR2 processing error: {e}")
                    result['processing_error'] = str(e)
            
            # Log performance metrics
            end_time = time.time()
            memory_used, peak_memory, peak_percentage = tracker.end()
            logger.info(f"Processed Canon {file_ext} data with {len(result)} attributes in {end_time - start_time:.2f} seconds")
            logger.info(f"Memory usage: {memory_used/(1024*1024):.2f} MB, Peak: {peak_memory/(1024*1024):.2f} MB ({peak_percentage:.1f}%)")
            
            return result
    
    def get_makernote_tags(self) -> Dict[str, str]:
        """Get Canon MakerNote tag mapping
        
        Returns:
            Dictionary mapping Canon MakerNote tag names to field names
        """
        return {
            'MakerNote CanonCameraSettings': 'canon_camera_settings',
            'MakerNote CanonFocalLength': 'canon_focal_length',
            'MakerNote CanonShotInfo': 'canon_shot_info',
            'MakerNote CanonFileInfo': 'canon_file_info',
            'MakerNote CanonImageType': 'canon_image_type',
            'MakerNote CanonFirmwareVersion': 'canon_firmware_version',
            'MakerNote CanonModelID': 'canon_model_id',
            'MakerNote CanonAFInfo': 'canon_af_info',
            'MakerNote CanonFlashInfo': 'canon_flash_info',
            'MakerNote CanonLensModel': 'canon_lens_model',
            'MakerNote CanonLensInfo': 'canon_lens_info',
            'MakerNote CanonWhiteBalance': 'canon_white_balance',
            'MakerNote CanonColorSpace': 'canon_color_space',
            'MakerNote CanonColorData': 'canon_color_data',
            'MakerNote CanonProcessingInfo': 'canon_processing_info',
            'MakerNote CanonToneCurve': 'canon_tone_curve',
            'MakerNote CanonColorInfo': 'canon_color_info',
            'MakerNote CanonPictureStyle': 'canon_picture_style',
            'MakerNote CanonCustomFunctions': 'canon_custom_functions'
        }
    
    def process_thumbnail(self, thumb_data: bytes, thumb_format: str) -> Optional[Dict[str, Any]]:
        """Process Canon thumbnail data with GPU acceleration if available
        
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
    
    def _check_exiftool(self) -> bool:
        """Check if exiftool is available
        
        Returns:
            True if exiftool is available, False otherwise
        """
        try:
            result = subprocess.run(['which', 'exiftool'], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True)
            return result.returncode == 0
        except Exception:
            return False
    
    def _run_exiftool(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Run exiftool to extract metadata
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing exiftool metadata, or None
        """
        try:
            result = subprocess.run(['exiftool', '-json', image_path], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True)
            if result.returncode == 0 and result.stdout:
                data = json.loads(result.stdout)
                if data and isinstance(data, list) and len(data) > 0:
                    return data[0]
        except Exception as e:
            print(f"Error running exiftool: {e}")
        return None
    
    def _extract_preview(self, image_path: str) -> Optional[str]:
        """Extract preview image from CR3 file using exiftool
        
        Args:
            image_path: Path to the CR3 file
            
        Returns:
            Path to the extracted preview image, or None
        """
        try:
            # Create a temporary file for the preview
            fd, temp_path = tempfile.mkstemp(suffix='.jpg')
            os.close(fd)
            
            # Extract preview image
            result = subprocess.run(['exiftool', '-b', '-PreviewImage', '-w', temp_path, image_path], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True)
            
            if result.returncode == 0 and os.path.exists(temp_path):
                return temp_path
        except Exception as e:
            print(f"Error extracting preview: {e}")
        return None
    
    def _process_preview_with_gpu(self, preview: Image.Image) -> Optional[Dict[str, Any]]:
        """Process preview image with GPU acceleration
        
        Args:
            preview: PIL Image of the preview
            
        Returns:
            Dictionary containing processed preview data, or None
        """
        result = {}
        
        try:
            import torch
            if torch.backends.mps.is_available():
                print("Processing preview with Metal GPU acceleration")
                # Convert preview to tensor and process with Metal
                preview_array = np.array(preview)
                preview_tensor = torch.from_numpy(preview_array).to('mps')
                
                # Calculate basic statistics
                result['canon_preview_mean_r'] = float(torch.mean(preview_tensor[:,:,0]).item())
                result['canon_preview_mean_g'] = float(torch.mean(preview_tensor[:,:,1]).item())
                result['canon_preview_mean_b'] = float(torch.mean(preview_tensor[:,:,2]).item())
                
                # Calculate histogram for each channel
                r_hist = torch.histc(preview_tensor[:,:,0].float(), bins=256, min=0, max=255)
                g_hist = torch.histc(preview_tensor[:,:,1].float(), bins=256, min=0, max=255)
                b_hist = torch.histc(preview_tensor[:,:,2].float(), bins=256, min=0, max=255)
                
                # Calculate dynamic range
                r_min, r_max = torch.min(preview_tensor[:,:,0]), torch.max(preview_tensor[:,:,0])
                g_min, g_max = torch.min(preview_tensor[:,:,1]), torch.max(preview_tensor[:,:,1])
                b_min, b_max = torch.min(preview_tensor[:,:,2]), torch.max(preview_tensor[:,:,2])
                
                result['canon_preview_dynamic_range_r'] = float(torch.log2(r_max - r_min + 1).item())
                result['canon_preview_dynamic_range_g'] = float(torch.log2(g_max - g_min + 1).item())
                result['canon_preview_dynamic_range_b'] = float(torch.log2(b_max - b_min + 1).item())
                
                print("Preview processing with GPU completed")
        except Exception as e:
            print(f"Error processing preview with GPU: {e}")
        
        return result
