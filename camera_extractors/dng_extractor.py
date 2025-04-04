#!/usr/bin/env python3
"""
DNG Camera Extractor
Provides DNG-specific EXIF extraction with special handling for Leica cameras
"""

import os
import io
import numpy as np
import rawpy
from PIL import Image
from typing import Dict, Any, Optional, List, Tuple
import subprocess
import json

from .base_extractor import CameraExtractor


class DngExtractor(CameraExtractor):
    """DNG-specific EXIF extractor with special handling for Leica cameras"""
    
    def can_handle(self, file_ext: str, exif_data: Dict[str, Any]) -> bool:
        """Check if this extractor can handle the given file
        
        Args:
            file_ext: File extension (e.g., '.dng')
            exif_data: Basic EXIF data already extracted
            
        Returns:
            True if this is a DNG file, False otherwise
        """
        # Check if it's a DNG file
        is_dng = file_ext.lower() == '.dng'
        
        # Special handling for Leica cameras
        if is_dng and exif_data.get('camera_make', '').upper().startswith('LEICA'):
            print("Detected Leica DNG file")
            self.is_leica = True
        else:
            self.is_leica = False
            
        return is_dng
    
    def extract_metadata(self, image_path: str, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract DNG-specific metadata from the image
        
        Args:
            image_path: Path to the image file
            exif_data: Basic EXIF data already extracted
            
        Returns:
            Dictionary containing DNG-specific metadata
        """
        result = {}
        
        # Try to extract more metadata using exiftool if available
        try:
            if self._check_exiftool():
                print("Using exiftool to extract DNG metadata")
                exiftool_data = self._run_exiftool(image_path)
                if exiftool_data:
                    # Process DNG-specific tags
                    for key, value in exiftool_data.items():
                        if key.startswith('DNG'):
                            # Convert to snake_case
                            dng_key = 'dng_' + key[3:].lower().replace(' ', '_')
                            result[dng_key] = value
                    
                    # Process Leica-specific tags if it's a Leica camera
                    if self.is_leica:
                        for key, value in exiftool_data.items():
                            if key.startswith('Leica'):
                                # Convert to snake_case
                                leica_key = 'leica_' + key[5:].lower().replace(' ', '_')
                                result[leica_key] = value
                    
                    # Extract GPS data if available
                    if 'GPSLatitude' in exiftool_data and 'GPSLongitude' in exiftool_data:
                        try:
                            # Convert GPS coordinates to decimal format
                            lat = self._parse_gps_coordinate(exiftool_data.get('GPSLatitude', '0'))
                            lon = self._parse_gps_coordinate(exiftool_data.get('GPSLongitude', '0'))
                            
                            # Apply N/S and E/W reference
                            if exiftool_data.get('GPSLatitudeRef', 'N') == 'S':
                                lat = -lat
                            if exiftool_data.get('GPSLongitudeRef', 'E') == 'W':
                                lon = -lon
                                
                            result['gps_latitude_dec'] = lat
                            result['gps_longitude_dec'] = lon
                            
                            # Add GPS altitude if available
                            if 'GPSAltitude' in exiftool_data:
                                alt = float(str(exiftool_data['GPSAltitude']).split(' ')[0])
                                if exiftool_data.get('GPSAltitudeRef', '0') == '1':  # Below sea level
                                    alt = -alt
                                result['gps_altitude_meters'] = alt
                        except Exception as gps_error:
                            print(f"Error parsing GPS data: {gps_error}")
        except Exception as e:
            print(f"Error extracting DNG metadata: {e}")
        
        return result
    
    def process_raw(self, image_path: str, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process DNG file data
        
        Args:
            image_path: Path to the DNG file
            exif_data: Basic EXIF data already extracted
            
        Returns:
            Dictionary containing processed DNG data
        """
        # Initialize result with basic file info regardless of processing success
        result = {
            'file_path': image_path,
            'file_size': os.path.getsize(image_path) if os.path.exists(image_path) else 0,
            'dng_model': exif_data.get('camera_model', 'Unknown'),
            'dng_make': exif_data.get('camera_make', 'Unknown'),
            'dng_opencv_compatible': True  # Default to True, will be set to False if issues detected
        }
        
        # First try to use exiftool as a safer alternative
        try:
            # Add specific handling for OpenCV errors with DNG files
            if 'opencv' in exif_data.get('processing_errors', []):
                print(f"Skipping OpenCV processing for DNG file due to known compatibility issues: {image_path}")
                result['dng_opencv_compatible'] = False
                result['processing_method'] = 'exiftool_only'
                # Still try to use exiftool for metadata
            
            if self._check_exiftool():
                print("Using exiftool for DNG metadata extraction")
                exiftool_data = self._run_exiftool(image_path)
                if exiftool_data:
                    # Extract key metadata from exiftool output
                    for key, value in exiftool_data.items():
                        if key.startswith('DNG'):
                            # Convert to snake_case
                            dng_key = 'dng_' + key[3:].lower().replace(' ', '_')
                            result[dng_key] = value
                    
                    # Extract Leica-specific data if applicable
                    if self.is_leica:
                        for key, value in exiftool_data.items():
                            if key.startswith('Leica'):
                                # Convert to snake_case
                                leica_key = 'leica_' + key[5:].lower().replace(' ', '_')
                                result[leica_key] = value
                    
                    # Add raw processing status
                    result['raw_processing_status'] = 'exiftool_primary'
        except Exception as e:
            print(f"Error using exiftool: {e}")
            result['exiftool_error'] = str(e)
        
        # Process DNG file with rawpy with robust error handling
        try:
            # Try to open the raw file with a thread-safe timeout approach
            import threading
            import concurrent.futures
            
            # Define a function to open the raw file
            def open_raw_file(path):
                try:
                    return rawpy.imread(path)
                except Exception as e:
                    print(f"Error in open_raw_file: {e}")
                    raise
            
            # Use a thread pool with a timeout instead of signals
            # This is thread-safe and works in any thread
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    # Submit the task and wait with a timeout
                    future = executor.submit(open_raw_file, image_path)
                    raw = future.result(timeout=10)  # Increased timeout to 10 seconds
                    print(f"Processing DNG file: {image_path}")
                    
                    try:
                        # Add specific handling for OpenCV-related operations
                        try:
                            # Check if raw has raw_image attribute
                            if not hasattr(raw, 'raw_image'):
                                print(f"Raw file does not have raw_image attribute: {image_path}")
                                result['dng_opencv_compatible'] = False
                                result['raw_image_error'] = "No raw_image attribute"
                                # Add basic information that doesn't require raw_image
                                if hasattr(raw, 'sizes'):
                                    result['dng_full_width'] = raw.sizes.width
                                    result['dng_full_height'] = raw.sizes.height
                                    print(f"DNG dimensions: {result['dng_full_width']}x{result['dng_full_height']}")
                                    result['raw_dimensions_extracted'] = True
                                raise ValueError("No raw_image attribute")
                                
                            # Get raw image data - use sampling to reduce memory usage
                            try:
                                raw_image = raw.raw_image.copy()
                            except Exception as copy_error:
                                print(f"Error copying raw image: {copy_error}")
                                # Try without copy
                                raw_image = raw.raw_image
                            
                            # For large images, sample to reduce memory usage
                            if raw_image.size > 20_000_000:  # For very large sensors
                                sample_rate = max(1, int(np.sqrt(raw_image.size / 1_000_000)))
                                try:
                                    sampled_image = raw_image[::sample_rate, ::sample_rate]
                                    print(f"Sampling raw image at 1/{sample_rate} for stats calculation")
                                except Exception as sampling_error:
                                    print(f"Error sampling image: {sampling_error}")
                                    # Use a smaller portion of the image
                                    try:
                                        max_size = min(1000, min(raw_image.shape))
                                        sampled_image = raw_image[:max_size, :max_size]
                                        print(f"Using first {max_size}x{max_size} pixels for analysis")
                                    except:
                                        # Last resort - just use a tiny slice
                                        sampled_image = raw_image[:100, :100]
                            else:
                                sampled_image = raw_image
                        except Exception as opencv_error:
                            # Handle OpenCV compatibility issues with DNG files
                            error_msg = str(opencv_error)
                            if 'opencv' in error_msg.lower() or 'imread_' in error_msg or 'original_ptr == real_mat.data' in error_msg:
                                print(f"OpenCV compatibility issue detected with DNG file: {image_path}")
                                print(f"Error details: {error_msg}")
                                result['dng_opencv_compatible'] = False
                                result['opencv_error'] = error_msg
                                # Skip further OpenCV processing
                                raise
                        
                        # Get basic stats about the raw image with error handling
                        try:
                            raw_min = np.min(sampled_image)
                            raw_max = np.max(sampled_image)
                            print(f"Raw image shape: {raw_image.shape}")
                            print(f"Raw image min/max values: {raw_min}/{raw_max}")
                            
                            # Calculate histogram with reduced bins for efficiency
                            try:
                                histogram, _ = np.histogram(sampled_image.flatten(), bins=256)
                                histogram_mean = float(np.mean(histogram))
                                histogram_std = float(np.std(histogram))
                            except Exception as hist_error:
                                print(f"Error calculating histogram: {hist_error}")
                                histogram_mean = 0
                                histogram_std = 0
                            
                            # DNG specific fields
                            dng_metadata = {
                                'dng_raw_image_shape': str(raw_image.shape),
                                'dng_raw_min_value': int(raw_min),
                                'dng_raw_max_value': int(raw_max),
                                'dng_raw_histogram_mean': histogram_mean,
                                'dng_raw_histogram_std': histogram_std,
                                'dng_raw_dynamic_range': float(np.log2(raw_max - raw_min + 1)) if raw_max > raw_min else 0,
                                'dng_raw_black_level': int(raw.black_level) if hasattr(raw, 'black_level') else 0,
                                'dng_raw_white_level': int(raw.white_level) if hasattr(raw, 'white_level') else 0
                            }
                        except Exception as stats_error:
                            print(f"Error calculating basic stats: {stats_error}")
                            # Provide minimal metadata
                            dng_metadata = {
                                'dng_raw_image_shape': str(raw_image.shape) if 'raw_image' in locals() else 'unknown',
                                'dng_raw_black_level': int(raw.black_level) if hasattr(raw, 'black_level') else 0,
                                'dng_raw_white_level': int(raw.white_level) if hasattr(raw, 'white_level') else 0
                            }
                        
                        # Add DNG metadata to result
                        result.update(dng_metadata)
                        result['raw_processing_status'] = 'rawpy_success'
                    except Exception as stats_error:
                        print(f"Error processing raw image stats: {stats_error}")
                        result['stats_error'] = str(stats_error)
                    
                    # Try to extract thumbnail - separate try block to ensure it runs even if stats fail
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
                        result['thumbnail_error'] = str(thumb_error)
                    
                    # Try to get color profile information - separate try block
                    try:
                        if hasattr(raw, 'color_desc'):
                            result['dng_color_profile'] = raw.color_desc.decode('utf-8', errors='ignore')
                            print(f"Color profile: {result['dng_color_profile']}")
                        
                        # Get white balance coefficients if available
                        if hasattr(raw, 'camera_whitebalance'):
                            # Handle different types of camera_whitebalance
                            if isinstance(raw.camera_whitebalance, np.ndarray) and hasattr(raw.camera_whitebalance, 'tolist'):
                                result['dng_camera_whitebalance'] = raw.camera_whitebalance.tolist()
                            else:
                                result['dng_camera_whitebalance'] = str(raw.camera_whitebalance)
                            print(f"Camera white balance: {result['dng_camera_whitebalance']}")
                        
                        # Get full resolution dimensions
                        if hasattr(raw, 'sizes'):
                            result['dng_full_width'] = raw.sizes.width
                            result['dng_full_height'] = raw.sizes.height
                            print(f"Full resolution: {result['dng_full_width']}x{result['dng_full_height']}")
                    except Exception as metadata_error:
                        print(f"Error extracting metadata: {metadata_error}")
                        result['metadata_error'] = str(metadata_error)
                    
                    # Add Leica-specific processing if needed
                    if self.is_leica and not any(k.endswith('_error') for k in result.keys()):
                        # Leica cameras often have specific color profiles
                        result['leica_processed'] = True
                        
                        # Use GPU acceleration for Leica files if available
                        if self.use_gpu:
                            try:
                                import torch
                                if torch.backends.mps.is_available():
                                    print("Processing Leica DNG with Metal GPU acceleration")
                                    # Convert raw data to tensor for processing
                                    raw_tensor = torch.from_numpy(sampled_image).to('mps')
                                    
                                    # Calculate basic statistics with GPU
                                    result['leica_raw_mean'] = float(torch.mean(raw_tensor).item())
                                    result['leica_raw_std'] = float(torch.std(raw_tensor).item())
                                    
                                    print("Leica GPU processing completed")
                            except Exception as gpu_error:
                                print(f"GPU processing error: {gpu_error}")
                                result['gpu_error'] = str(gpu_error)
            except concurrent.futures.TimeoutError as e:
                print(f"Timeout opening DNG file: {e}")
                result['rawpy_timeout_error'] = str(e)
                result['raw_processing_status'] = 'timeout'
            except Exception as future_error:
                print(f"Error in future execution: {future_error}")
                result['future_error'] = str(future_error)
                result['raw_processing_status'] = 'future_error'
            except (rawpy.LibRawError, ValueError, IOError) as e:
                print(f"Error opening DNG file with rawpy: {e}")
                result['rawpy_error'] = str(e)
                result['raw_processing_status'] = 'failed'
            finally:
                # No signal handlers to restore in this approach
                pass
        except ImportError:
            print("rawpy not available for DNG processing")
            result['rawpy_import_error'] = "rawpy module not available"
        except Exception as e:
            print(f"DNG processing error: {e}")
            result['processing_error'] = str(e)
        
        return result
    
    def get_makernote_tags(self) -> Dict[str, str]:
        """Get DNG MakerNote tag mapping
        
        Returns:
            Dictionary mapping DNG MakerNote tag names to field names
        """
        makernote_tags = {
            'MakerNote DNGVersion': 'dng_version',
            'MakerNote DNGBackwardVersion': 'dng_backward_version',
            'MakerNote DNGPrivateData': 'dng_private_data',
            'MakerNote ColorimetricReference': 'dng_colorimetric_reference',
            'MakerNote AsShotICCProfile': 'dng_as_shot_icc_profile',
            'MakerNote AsShotPreProfileMatrix': 'dng_as_shot_pre_profile_matrix',
            'MakerNote CurrentICCProfile': 'dng_current_icc_profile',
            'MakerNote CurrentPreProfileMatrix': 'dng_current_pre_profile_matrix'
        }
        
        # Add Leica-specific tags if needed
        if self.is_leica:
            leica_tags = {
                'MakerNote LeicaLensID': 'leica_lens_id',
                'MakerNote LeicaLensType': 'leica_lens_type',
                'MakerNote LeicaFirmware': 'leica_firmware',
                'MakerNote LeicaSerialNumber': 'leica_serial_number',
                'MakerNote LeicaFocalLength': 'leica_focal_length',
                'MakerNote LeicaAperture': 'leica_aperture',
                'MakerNote LeicaExposureMode': 'leica_exposure_mode',
                'MakerNote LeicaExposureProgram': 'leica_exposure_program',
                'MakerNote LeicaMeteringMode': 'leica_metering_mode',
                'MakerNote LeicaWhiteBalance': 'leica_white_balance',
                'MakerNote LeicaImageQuality': 'leica_image_quality'
            }
            makernote_tags.update(leica_tags)
        
        return makernote_tags
    
    def process_thumbnail(self, thumb_data: bytes, thumb_format: str) -> Optional[Dict[str, Any]]:
        """Process DNG thumbnail data with GPU acceleration if available
        
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
    
    def _parse_gps_coordinate(self, coord_str: str) -> float:
        """Parse GPS coordinate string to decimal format
        
        Args:
            coord_str: GPS coordinate string (e.g., '40 deg 26' 46.56" N')
            
        Returns:
            GPS coordinate in decimal format
        """
        # Handle different formats
        if 'deg' in coord_str:
            # Format: '40 deg 26' 46.56" N'
            parts = coord_str.replace('deg', '').replace("'", '').replace('"', '').split()
            degrees = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2]) if len(parts) > 2 else 0
        else:
            # Format: '40.446267'
            try:
                return float(coord_str)
            except ValueError:
                # Format: '40 26 46.56'
                parts = coord_str.split()
                degrees = float(parts[0])
                minutes = float(parts[1]) if len(parts) > 1 else 0
                seconds = float(parts[2]) if len(parts) > 2 else 0
        
        # Convert to decimal format
        return degrees + (minutes / 60.0) + (seconds / 3600.0)
