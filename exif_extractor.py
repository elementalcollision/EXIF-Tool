#!/usr/bin/env python3
"""
Enhanced EXIF extractor module for the EXIF tool
This module provides improved EXIF extraction capabilities
with modular support for different camera types
"""

import os
import sys
import exifread
import datetime
import time
import io
import warnings
import numpy as np
import rawpy
from PIL import Image, ImageFile
from camera_extractors import get_camera_extractor

# Disable DecompressionBombWarning
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

class EnhancedExifExtractor:
    """Enhanced EXIF extractor with support for Sony ARW files"""
    
    def __init__(self, use_gpu=False, cpu_cores=None, memory_limit_percent=75):
        """Initialize the extractor with resource management settings
        
        Args:
            use_gpu (bool): Whether to use GPU acceleration when available
            cpu_cores (int): Number of CPU cores to use, defaults to n-2 on Apple Silicon
            memory_limit_percent (int): Maximum memory usage as percentage of total RAM
        """
        # Resource management settings
        self.use_gpu = use_gpu
        
        # Set CPU cores to use (n-2 for ARM cores on Apple Silicon)
        import multiprocessing
        total_cores = multiprocessing.cpu_count()
        if cpu_cores is None:
            # Default to n-2 cores on ARM processors (Apple Silicon)
            import platform
            if platform.processor() == 'arm':
                self.cpu_cores = max(1, total_cores - 2)
            else:
                self.cpu_cores = max(1, total_cores - 1)
        else:
            self.cpu_cores = min(max(1, cpu_cores), total_cores)
        
        # Set memory limit
        import psutil
        self.total_memory = psutil.virtual_memory().total
        self.memory_limit = self.total_memory * (memory_limit_percent / 100.0)
        
        # Initialize GPU if available and requested
        if self.use_gpu:
            try:
                # Check for Metal GPU support on Apple Silicon
                import torch
                self.has_mps = torch.backends.mps.is_available()
                if self.has_mps:
                    print(f"Metal GPU acceleration enabled on Apple Silicon")
                    self.device = torch.device("mps")
                else:
                    print("Metal GPU acceleration not available")
                    self.use_gpu = False
            except ImportError:
                print("PyTorch not available, GPU acceleration disabled")
                self.use_gpu = False
        
        print(f"EnhancedExifExtractor initialized with {self.cpu_cores} CPU cores, "
              f"memory limit: {memory_limit_percent}% ({self.memory_limit / (1024**3):.1f} GB), "
              f"GPU acceleration: {self.use_gpu}")
              
        # Set supported file extensions
        self.supported_extensions = ['.jpg', '.jpeg', '.tiff', '.tif', '.png', '.heic', '.heif', '.nef', '.cr2', '.cr3', '.arw', '.dng', '.raf']
    
    def extract_exif(self, image_path):
        """Extract EXIF data from an image file using multiple methods"""
        print(f"\nExtracting EXIF from: {image_path}")
        
        # Check if file exists and is accessible
        if not os.path.isfile(image_path) or not os.access(image_path, os.R_OK):
            print(f"File not accessible: {image_path}")
            return None
        
        # Get basic file info
        file_size = os.path.getsize(image_path)
        file_ext = os.path.splitext(image_path)[1].lower()
        file_name = os.path.basename(image_path)
        
        print(f"File: {file_name} ({file_ext})")
        print(f"Size: {file_size / 1024:.1f} KB")
        
        # Initialize result dictionary with basic file info
        result = {
            'file_path': image_path,
            'file_name': file_name,
            'file_size': file_size,
            'file_type': file_ext.lstrip('.').upper(),
            'date_processed': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Try to get file creation/modification dates
        try:
            file_stat = os.stat(image_path)
            file_created = datetime.datetime.fromtimestamp(file_stat.st_ctime)
            file_modified = datetime.datetime.fromtimestamp(file_stat.st_mtime)
            
            result['file_created'] = file_created.strftime('%Y-%m-%d %H:%M:%S')
            result['file_modified'] = file_modified.strftime('%Y-%m-%d %H:%M:%S')
            
            # Use file dates as fallback for date_taken if not found later
            result['date_taken'] = file_modified.strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            print(f"Error getting file dates: {e}")
        
        # Method 1: PIL for basic image properties and some EXIF (works for JPEG, TIFF, etc.)
        # For RAW files like ARW, this will likely fail, and we'll use other methods
        pil_success = False
        try:
            with Image.open(image_path) as img:
                # Get basic image properties
                width, height = img.size
                result['width'] = width
                result['height'] = height
                result['aspect_ratio'] = round(width / height, 2) if height > 0 else 0
                result['format'] = img.format or 'UNKNOWN'
                pil_success = True
                
                print(f"Image: {width}x{height} {result['format']}")
                
                # Try to get EXIF data
                if hasattr(img, '_getexif') and img._getexif():
                    exif = img._getexif()
                    print(f"Found {len(exif)} EXIF tags with PIL")
                    
                    # Map common EXIF tags
                    exif_tags = {
                        271: 'camera_make',      # Make
                        272: 'camera_model',     # Model
                        306: 'date_modified',    # DateTime
                        36867: 'date_taken',     # DateTimeOriginal
                        33434: 'exposure_time',  # ExposureTime
                        33437: 'f_number',       # FNumber
                        34855: 'iso',            # ISOSpeedRatings
                        37386: 'focal_length',   # FocalLength
                        274: 'orientation',      # Orientation
                        305: 'software',         # Software
                        315: 'artist',           # Artist
                        36868: 'date_digitized', # DateTimeDigitized
                        37520: 'subsec_time',    # SubSecTime
                        37377: 'shutter_speed',  # ShutterSpeedValue
                        37378: 'aperture',       # ApertureValue
                        37380: 'exposure_comp',  # ExposureBiasValue
                        37383: 'metering_mode',  # MeteringMode
                        37384: 'light_source',   # LightSource
                        37385: 'flash',          # Flash
                    }
                    
                    # Extract values
                    for tag_id, field in exif_tags.items():
                        if tag_id in exif:
                            result[field] = str(exif[tag_id])
                else:
                    print("No EXIF data found with PIL")
        except Exception as e:
            print(f"PIL error: {e}")
        
        # Method 2: ExifRead for more detailed EXIF data (works well with RAW files)
        try:
            with open(image_path, 'rb') as f:
                # Use details=True for Sony ARW files to get all metadata
                tags = exifread.process_file(f, details=True, strict=False)
                
                if tags:
                    print(f"Found {len(tags)} tags with ExifRead")
                    
                    # Map common tags - expanded for Sony ARW files
                    tag_mapping = {
                        # Basic image info
                        'Image Make': 'camera_make',
                        'Image Model': 'camera_model',
                        'Image Software': 'software',
                        'Image DateTime': 'datetime',
                        'Image Orientation': 'orientation',
                        'Image XResolution': 'x_resolution',
                        'Image YResolution': 'y_resolution',
                        
                        # EXIF standard tags
                        'EXIF DateTimeOriginal': 'date_taken',
                        'EXIF DateTimeDigitized': 'date_digitized',
                        'EXIF ExposureTime': 'exposure_time',
                        'EXIF FNumber': 'f_number',
                        'EXIF ExposureProgram': 'exposure_program',
                        'EXIF ISOSpeedRatings': 'iso',
                        'EXIF SubjectDistance': 'subject_distance',
                        'EXIF MeteringMode': 'metering_mode',
                        'EXIF LightSource': 'light_source',
                        'EXIF Flash': 'flash',
                        'EXIF FocalLength': 'focal_length',
                        'EXIF SubjectArea': 'subject_area',
                        'EXIF FlashEnergy': 'flash_energy',
                        'EXIF FocalPlaneXResolution': 'focal_plane_x_resolution',
                        'EXIF FocalPlaneYResolution': 'focal_plane_y_resolution',
                        'EXIF ExposureMode': 'exposure_mode',
                        'EXIF WhiteBalance': 'white_balance',
                        'EXIF DigitalZoomRatio': 'digital_zoom_ratio',
                        'EXIF SceneCaptureType': 'scene_capture_type',
                        'EXIF GainControl': 'gain_control',
                        'EXIF Contrast': 'contrast',
                        'EXIF Saturation': 'saturation',
                        'EXIF Sharpness': 'sharpness',
                        
                        # Lens information
                        'EXIF LensSpecification': 'lens_specification',
                        'EXIF LensMake': 'lens_make',
                        'EXIF LensModel': 'lens_model',
                        'EXIF LensSerialNumber': 'lens_serial_number',
                        
                        # GPS data
                        'GPS GPSLatitude': 'gps_latitude',
                        'GPS GPSLongitude': 'gps_longitude',
                        'GPS GPSAltitude': 'gps_altitude',
                        'GPS GPSTimeStamp': 'gps_timestamp',
                        'GPS GPSDateStamp': 'gps_datestamp',
                        
                        # Sony specific tags
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
                    
                    # Extract values
                    for tag, field in tag_mapping.items():
                        if tag in tags:
                            result[field] = str(tags[tag])
                    
                    # If PIL failed but ExifRead succeeded, try to get dimensions from ExifRead
                    if not pil_success and 'EXIF ExifImageWidth' in tags and 'EXIF ExifImageLength' in tags:
                        try:
                            result['width'] = int(str(tags['EXIF ExifImageWidth']))
                            result['height'] = int(str(tags['EXIF ExifImageLength']))
                            result['aspect_ratio'] = round(result['width'] / result['height'], 2) if result['height'] > 0 else 0
                            print(f"Image dimensions from ExifRead: {result['width']}x{result['height']}")
                        except:
                            pass
                    
                    # For Sony ARW files, also capture any MakerNote tags that might be useful
                    if file_ext.lower() == '.arw' and 'camera_make' in result and 'SONY' in result['camera_make'].upper():
                        print("Processing Sony-specific MakerNote tags")
                        sony_tags = {k: str(v) for k, v in tags.items() if 'MakerNote' in k}
                        if sony_tags:
                            result['has_sony_makernote'] = True
                            result['sony_makernote_count'] = len(sony_tags)
                            
                            # Add a few important Sony tags directly
                            for tag_name, tag_value in sony_tags.items():
                                # Clean up the tag name to make it more readable
                                clean_name = tag_name.replace('MakerNote ', '').lower().replace(' ', '_')
                                field_name = f"sony_{clean_name}"
                                result[field_name] = str(tag_value)
                else:
                    print("No tags found with ExifRead")
        except Exception as e:
            print(f"ExifRead error: {e}")
        
        # Method 3: RAW processing for RAW files using camera-specific extractors
        if file_ext.lower() in ['.arw', '.nef', '.cr2', '.cr3', '.orf', '.rw2', '.raw', '.dng', '.raf']:
            try:
                print(f"Processing RAW file: {file_ext}")
                
                # First, try to get a camera-specific extractor
                camera_extractor = get_camera_extractor(
                    file_ext=file_ext,
                    exif_data=result,
                    use_gpu=self.use_gpu,
                    memory_limit=self.memory_limit / self.total_memory,
                    cpu_cores=self.cpu_cores
                )
                
                # If we have a camera-specific extractor, use it
                if camera_extractor:
                    # Extract camera-specific metadata
                    camera_metadata = camera_extractor.extract_metadata(image_path, result)
                    if camera_metadata:
                        result.update(camera_metadata)
                    
                    # Process RAW data
                    raw_data = camera_extractor.process_raw(image_path, result)
                    if raw_data:
                        result.update(raw_data)
                    
                    # Add any MakerNote tags to the tag mapping
                    makernote_tags = camera_extractor.get_makernote_tags()
                    if makernote_tags and tags:  # tags from exifread
                        for tag, field in makernote_tags.items():
                            if tag in tags:
                                result[field] = str(tags[tag])
                
                # If no camera-specific extractor or as a fallback, use generic RAW processing
                else:
                    with rawpy.imread(image_path) as raw:
                        # Extract basic RAW metadata
                        raw_metadata = {
                            'raw_type': str(raw.raw_type),
                            'raw_pattern': str(raw.raw_pattern.tolist()) if hasattr(raw, 'raw_pattern') else None,
                            'black_level': str(raw.black_level) if hasattr(raw, 'black_level') else None,
                            'white_level': str(raw.white_level) if hasattr(raw, 'white_level') else None,
                            'color_desc': raw.color_desc.decode('utf-8', errors='ignore') if hasattr(raw, 'color_desc') else None,
                            'num_colors': raw.num_colors if hasattr(raw, 'num_colors') else None,
                            'raw_height': raw.sizes.raw_height if hasattr(raw, 'sizes') else None,
                            'raw_width': raw.sizes.raw_width if hasattr(raw, 'sizes') else None
                        }
                        
                        # Update dimensions if not already set
                        if 'width' not in result or result['width'] == 0:
                            result['width'] = raw.sizes.width if hasattr(raw, 'sizes') else 0
                            result['height'] = raw.sizes.height if hasattr(raw, 'sizes') else 0
                            result['aspect_ratio'] = round(result['width'] / result['height'], 2) if result['height'] > 0 else 0
                            print(f"RAW dimensions: {result['width']}x{result['height']}")
                        
                        # Add RAW metadata to result
                        for key, value in raw_metadata.items():
                            if value is not None:
                                result[key] = value
                        
                        # Try to extract thumbnail
                        try:
                            thumb = raw.extract_thumb()
                            if thumb and hasattr(thumb, 'format'):
                                result['has_thumbnail'] = True
                                result['thumbnail_format'] = thumb.format
                                print(f"Extracted thumbnail in {thumb.format} format")
                        except Exception as thumb_error:
                            result['has_thumbnail'] = False
                            print(f"Thumbnail extraction error: {thumb_error}")
            except Exception as raw_error:
                print(f"RAW processing error: {raw_error}")
        
        # Add camera info if available
        if 'camera_make' in result and 'camera_model' in result:
            result['camera'] = f"{result['camera_make']} {result['camera_model']}"
        elif 'camera_make' in result:
            result['camera'] = result['camera_make']
        elif 'camera_model' in result:
            result['camera'] = result['camera_model']
        
        print(f"Extracted {len(result)} fields")
        return result

# Test function
def test_extractor(image_path, use_gpu=False):
    """Test the extractor with a single image
    
    Args:
        image_path (str): Path to the image file
        use_gpu (bool): Whether to use GPU acceleration
    """
    # Create extractor with resource management settings for Apple Silicon
    extractor = EnhancedExifExtractor(
        use_gpu=use_gpu,       # Use GPU acceleration if available
        cpu_cores=None,        # Auto-detect (n-2 on Apple Silicon)
        memory_limit_percent=75  # Cap memory usage at 75% of available RAM
    )
    
    # Extract EXIF data
    result = extractor.extract_exif(image_path)
    
    if result:
        print("\nExtracted EXIF data:")
        # Print the most important fields first
        priority_fields = [
            'file_name', 'camera_make', 'camera_model', 'date_taken', 
            'exposure_time', 'f_number', 'iso', 'focal_length',
            'width', 'height', 'sony_arw_version', 'sony_raw_black_level', 'sony_raw_white_level'
        ]
        
        # Print priority fields first
        for field in priority_fields:
            if field in result:
                print(f"{field}: {result[field]}")
        
        # Print remaining fields
        print("\nAdditional metadata:")
        for key, value in sorted(result.items()):
            if key not in priority_fields:
                print(f"{key}: {value}")
    else:
        print("Failed to extract EXIF data")

if __name__ == "__main__":
    import argparse
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Enhanced EXIF Extractor')
    parser.add_argument('image_path', nargs='?', help='Path to the image file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration if available')
    parser.add_argument('--cores', type=int, help='Number of CPU cores to use (default: n-2 on Apple Silicon)')
    parser.add_argument('--memory', type=int, default=75, help='Memory limit as percentage of total RAM (default: 75)')
    
    args = parser.parse_args()
    
    # If a path is provided, use it
    if args.image_path:
        test_extractor(args.image_path, use_gpu=args.gpu)
    else:
        print("Usage: python exif_extractor.py <image_path> [--gpu] [--cores N] [--memory PERCENT]")
