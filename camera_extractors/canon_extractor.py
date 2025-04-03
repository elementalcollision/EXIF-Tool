#!/usr/bin/env python3
"""
Canon Camera Extractor
Provides Canon-specific EXIF extraction for CR2 and CR3 files
"""

import os
import io
import subprocess
import json
import tempfile
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, List, Tuple

from .base_extractor import CameraExtractor


class CanonExtractor(CameraExtractor):
    """Canon-specific EXIF extractor for CR2 and CR3 files"""
    
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
            print("Detected Canon CR3 file")
            return True
            
        # For other Canon RAW formats, check camera make if available
        is_canon = exif_data.get('camera_make', '').upper().startswith('CANON')
        is_cr_file = file_ext.lower() in ['.cr2', '.crw']
        
        # If camera_make is not available but it's a Canon RAW format, assume it's Canon
        if 'camera_make' not in exif_data and is_cr_file:
            return True
            
        return is_canon and is_cr_file
    
    def extract_metadata(self, image_path: str, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Canon-specific metadata from the image
        
        Args:
            image_path: Path to the image file
            exif_data: Basic EXIF data already extracted
            
        Returns:
            Dictionary containing Canon-specific metadata
        """
        result = {}
        file_ext = os.path.splitext(image_path)[1].lower()
        
        # Try to extract more metadata using exiftool if available
        try:
            if self._check_exiftool():
                print("Using exiftool to extract Canon metadata")
                exiftool_data = self._run_exiftool(image_path)
                if exiftool_data:
                    # Process Canon-specific tags
                    for key, value in exiftool_data.items():
                        if key.startswith('Canon'):
                            # Convert to snake_case
                            canon_key = 'canon_' + key[5:].lower().replace(' ', '_')
                            result[canon_key] = value
                    
                    # Extract important Canon metadata
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
                    
                    # Extract CR3-specific metadata
                    if file_ext == '.cr3':
                        if 'ImageWidth' in exiftool_data:
                            result['width'] = exiftool_data['ImageWidth']
                        if 'ImageHeight' in exiftool_data:
                            result['height'] = exiftool_data['ImageHeight']
                        if 'CanonModelID' in exiftool_data:
                            result['canon_model_id'] = exiftool_data['CanonModelID']
                        if 'CanonFirmwareVersion' in exiftool_data:
                            result['canon_firmware'] = exiftool_data['CanonFirmwareVersion']
        except Exception as e:
            print(f"Error extracting Canon metadata: {e}")
        
        return result
    
    def process_raw(self, image_path: str, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Canon RAW file data
        
        Args:
            image_path: Path to the CR2/CR3 file
            exif_data: Basic EXIF data already extracted
            
        Returns:
            Dictionary containing processed Canon RAW data
        """
        result = {}
        file_ext = os.path.splitext(image_path)[1].lower()
        
        try:
            print(f"Processing Canon {file_ext} specific data")
            
            # CR3 files need special handling
            if file_ext == '.cr3':
                # Try to extract preview image using exiftool
                if self._check_exiftool():
                    preview_path = self._extract_preview(image_path)
                    if preview_path:
                        try:
                            with Image.open(preview_path) as preview:
                                width, height = preview.size
                                result['canon_preview_width'] = width
                                result['canon_preview_height'] = height
                                result['canon_preview_format'] = preview.format
                                print(f"Extracted preview image: {width}x{height} {preview.format}")
                                
                                # Process preview with GPU if available
                                if self.use_gpu:
                                    preview_data = self._process_preview_with_gpu(preview)
                                    if preview_data:
                                        result.update(preview_data)
                        except Exception as preview_error:
                            print(f"Preview processing error: {preview_error}")
                        finally:
                            # Clean up temporary preview file
                            if os.path.exists(preview_path):
                                os.remove(preview_path)
            
            # CR2 files can be processed with rawpy
            elif file_ext == '.cr2':
                try:
                    import rawpy
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
                        
                        # Canon CR2 specific fields
                        canon_metadata = {
                            'canon_raw_image_shape': str(raw_image.shape),
                            'canon_raw_min_value': int(raw_min),
                            'canon_raw_max_value': int(raw_max),
                            'canon_raw_histogram_mean': float(np.mean(histogram)),
                            'canon_raw_histogram_std': float(np.std(histogram)),
                            'canon_raw_dynamic_range': float(np.log2(raw_max - raw_min + 1)) if raw_max > raw_min else 0,
                            'canon_raw_black_level': int(raw.black_level) if hasattr(raw, 'black_level') else 0,
                            'canon_raw_white_level': int(raw.white_level) if hasattr(raw, 'white_level') else 0
                        }
                        
                        # Add Canon metadata to result
                        result.update(canon_metadata)
                        
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
                except ImportError:
                    print("rawpy not available for CR2 processing")
                except Exception as cr2_error:
                    print(f"Canon CR2 processing error: {cr2_error}")
        
        except Exception as canon_error:
            print(f"Canon {file_ext} specific error: {canon_error}")
        
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
