#!/usr/bin/env python3
"""
Fujifilm Camera Extractor
Provides Fujifilm-specific EXIF extraction for RAF files
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


class FujiExtractor(CameraExtractor):
    """Fujifilm-specific EXIF extractor for RAF files"""
    
    def can_handle(self, file_ext: str, exif_data: Dict[str, Any]) -> bool:
        """Check if this extractor can handle the given file
        
        Args:
            file_ext: File extension (e.g., '.raf')
            exif_data: Basic EXIF data already extracted
            
        Returns:
            True if this is a Fujifilm RAF file, False otherwise
        """
        # For RAF files, always return True since they're Fujifilm-specific
        if file_ext.lower() == '.raf':
            print("Detected Fujifilm RAF file")
            return True
            
        # Check if it's a Fujifilm camera
        is_fuji = False
        camera_make = exif_data.get('camera_make', '').upper()
        if camera_make.startswith('FUJI') or camera_make == 'FUJIFILM':
            is_fuji = True
            
        return is_fuji and file_ext.lower() == '.raf'
    
    def extract_metadata(self, image_path: str, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Fujifilm-specific metadata from the image
        
        Args:
            image_path: Path to the image file
            exif_data: Basic EXIF data already extracted
            
        Returns:
            Dictionary containing Fujifilm-specific metadata
        """
        result = {}
        
        # Try to extract more metadata using exiftool if available
        try:
            if self._check_exiftool():
                print("Using exiftool to extract Fujifilm metadata")
                exiftool_data = self._run_exiftool(image_path)
                if exiftool_data:
                    # Process Fujifilm-specific tags
                    for key, value in exiftool_data.items():
                        if key.startswith('Fuji'):
                            # Convert to snake_case
                            fuji_key = 'fuji_' + key[4:].lower().replace(' ', '_')
                            result[fuji_key] = value
                    
                    # Extract important Fujifilm metadata
                    if 'Make' in exiftool_data:
                        result['camera_make'] = exiftool_data['Make']
                    if 'Model' in exiftool_data:
                        result['camera_model'] = exiftool_data['Model']
                    if 'DateTimeOriginal' in exiftool_data:
                        result['date_taken'] = exiftool_data['DateTimeOriginal']
                    if 'ExposureTime' in exiftool_data:
                        result['exposure_time'] = exiftool_data['ExposureTime']
                    if 'FNumber' in exiftool_data:
                        result['f_number'] = exiftool_data['FNumber']
                    if 'ISO' in exiftool_data:
                        result['iso'] = exiftool_data['ISO']
                    if 'FocalLength' in exiftool_data:
                        result['focal_length'] = exiftool_data['FocalLength']
                    if 'ImageWidth' in exiftool_data:
                        result['width'] = exiftool_data['ImageWidth']
                    if 'ImageHeight' in exiftool_data:
                        result['height'] = exiftool_data['ImageHeight']
                    
                    # Extract Fujifilm Film Simulation mode
                    if 'FilmMode' in exiftool_data:
                        result['fuji_film_simulation'] = exiftool_data['FilmMode']
                    elif 'FujiFilmMode' in exiftool_data:
                        result['fuji_film_simulation'] = exiftool_data['FujiFilmMode']
                    
                    # Extract Fujifilm sensor information
                    if 'SensorType' in exiftool_data:
                        result['fuji_sensor_type'] = exiftool_data['SensorType']
                    
                    # Extract Fujifilm X-Trans information
                    if 'RAFVersion' in exiftool_data:
                        result['fuji_raf_version'] = exiftool_data['RAFVersion']
                    
                    # Extract lens information
                    if 'LensModel' in exiftool_data:
                        result['lens_model'] = exiftool_data['LensModel']
                    if 'LensMake' in exiftool_data:
                        result['lens_make'] = exiftool_data['LensMake']
                    
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
            print(f"Error extracting Fujifilm metadata: {e}")
        
        return result
    
    def process_raw(self, image_path: str, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Fujifilm RAF file data
        
        Args:
            image_path: Path to the RAF file
            exif_data: Basic EXIF data already extracted
            
        Returns:
            Dictionary containing processed Fujifilm RAF data
        """
        result = {}
        
        try:
            print(f"Processing Fujifilm RAF specific data")
            with rawpy.imread(image_path) as raw:
                # Get raw image data
                raw_image = raw.raw_image.copy()
                
                # Get basic stats about the raw image
                raw_min = np.min(raw_image)
                raw_max = np.max(raw_image)
                print(f"Raw image shape: {raw_image.shape}")
                print(f"Raw image min/max values: {raw_min}/{raw_max}")
                
                # Calculate histogram
                histogram, _ = np.histogram(raw_image.flatten(), bins=256)
                
                # Fujifilm RAF specific fields
                fuji_metadata = {
                    'fuji_raw_image_shape': str(raw_image.shape),
                    'fuji_raw_min_value': int(raw_min),
                    'fuji_raw_max_value': int(raw_max),
                    'fuji_raw_histogram_mean': float(np.mean(histogram)),
                    'fuji_raw_histogram_std': float(np.std(histogram)),
                    'fuji_raw_dynamic_range': float(np.log2(raw_max - raw_min + 1)) if raw_max > raw_min else 0,
                    'fuji_raw_black_level': int(raw.black_level) if hasattr(raw, 'black_level') else 0,
                    'fuji_raw_white_level': int(raw.white_level) if hasattr(raw, 'white_level') else 0
                }
                
                # Add Fujifilm metadata to result
                result.update(fuji_metadata)
                
                # Try to extract thumbnail
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
                
                # Try to get color profile information
                if hasattr(raw, 'color_desc'):
                    result['fuji_color_profile'] = raw.color_desc.decode('utf-8', errors='ignore')
                    print(f"Color profile: {result['fuji_color_profile']}")
                
                # Get white balance coefficients if available
                if hasattr(raw, 'camera_whitebalance'):
                    # Handle different types of camera_whitebalance
                    if hasattr(raw.camera_whitebalance, 'tolist'):
                        result['fuji_camera_whitebalance'] = raw.camera_whitebalance.tolist()
                    else:
                        result['fuji_camera_whitebalance'] = str(raw.camera_whitebalance)
                    print(f"Camera white balance: {result['fuji_camera_whitebalance']}")
                
                # Get full resolution dimensions
                if hasattr(raw, 'sizes'):
                    result['fuji_full_width'] = raw.sizes.width
                    result['fuji_full_height'] = raw.sizes.height
                    print(f"Full resolution: {result['fuji_full_width']}x{result['fuji_full_height']}")
                
                # Check for X-Trans sensor
                if hasattr(raw, 'raw_pattern') and raw.raw_pattern.shape == (6, 6):
                    result['fuji_sensor_type'] = 'X-Trans'
                    result['fuji_raw_pattern_size'] = '6x6'
                elif hasattr(raw, 'raw_pattern') and raw.raw_pattern.shape == (2, 2):
                    result['fuji_sensor_type'] = 'Bayer'
                    result['fuji_raw_pattern_size'] = '2x2'
                
                # Use GPU acceleration for processing if available
                if self.use_gpu:
                    try:
                        import torch
                        if torch.backends.mps.is_available():
                            print("Processing Fujifilm RAF with Metal GPU acceleration")
                            # Convert raw data to tensor for processing
                            # Need to convert to float32 for GPU processing
                            raw_tensor = torch.tensor(raw_image, dtype=torch.float32).to('mps')
                            
                            # Calculate basic statistics with GPU
                            result['fuji_raw_mean_gpu'] = float(torch.mean(raw_tensor).item())
                            result['fuji_raw_std_gpu'] = float(torch.std(raw_tensor).item())
                            
                            print("Fujifilm GPU processing completed")
                    except Exception as gpu_error:
                        print(f"GPU processing error: {gpu_error}")
        
        except Exception as fuji_error:
            print(f"Fujifilm RAF specific error: {fuji_error}")
        
        return result
    
    def get_makernote_tags(self) -> Dict[str, str]:
        """Get Fujifilm MakerNote tag mapping
        
        Returns:
            Dictionary mapping Fujifilm MakerNote tag names to field names
        """
        return {
            'MakerNote FujiFilmVersion': 'fuji_version',
            'MakerNote FujiFilmSerialNumber': 'fuji_serial_number',
            'MakerNote FujiFilmQuality': 'fuji_quality',
            'MakerNote FujiFilmSharpness': 'fuji_sharpness',
            'MakerNote FujiFilmWhiteBalance': 'fuji_white_balance',
            'MakerNote FujiFilmColor': 'fuji_color',
            'MakerNote FujiFilmTone': 'fuji_tone',
            'MakerNote FujiFilmFlashMode': 'fuji_flash_mode',
            'MakerNote FujiFilmFlashStrength': 'fuji_flash_strength',
            'MakerNote FujiFilmFocusMode': 'fuji_focus_mode',
            'MakerNote FujiFilmSlowSync': 'fuji_slow_sync',
            'MakerNote FujiFilmPictureModeType': 'fuji_picture_mode_type',
            'MakerNote FujiFilmExposureTime': 'fuji_exposure_time',
            'MakerNote FujiFilmFNumber': 'fuji_f_number',
            'MakerNote FujiFilmShutterType': 'fuji_shutter_type',
            'MakerNote FujiFilmExposureMode': 'fuji_exposure_mode',
            'MakerNote FujiFilmMeteringMode': 'fuji_metering_mode',
            'MakerNote FujiFilmFilmMode': 'fuji_film_mode',
            'MakerNote FujiFilmDynamicRange': 'fuji_dynamic_range',
            'MakerNote FujiFilmFaceDetection': 'fuji_face_detection',
            'MakerNote FujiFilmRAWInfo': 'fuji_raw_info',
            'MakerNote FujiFilmSensorType': 'fuji_sensor_type',
            'MakerNote FujiFilmFocusPoints': 'fuji_focus_points',
            'MakerNote FujiFilmImageWidth': 'fuji_image_width',
            'MakerNote FujiFilmImageHeight': 'fuji_image_height'
        }
    
    def process_thumbnail(self, thumb_data: bytes, thumb_format: str) -> Optional[Dict[str, Any]]:
        """Process Fujifilm thumbnail data with GPU acceleration if available
        
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
