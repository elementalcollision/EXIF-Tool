#!/usr/bin/env python3
"""
Camera Extractor Factory
Provides a factory pattern for creating camera-specific extractors
"""

from typing import Dict, Any, Type, Optional
from .base_extractor import CameraExtractor

# Registry of camera extractors
_CAMERA_EXTRACTORS = {}


def register_camera_extractor(camera_make: str, extractor_class: Type[CameraExtractor]) -> None:
    """Register a camera extractor for a specific camera make
    
    Args:
        camera_make: Camera manufacturer (e.g., 'SONY', 'NIKON')
        extractor_class: Class that implements CameraExtractor
    """
    _CAMERA_EXTRACTORS[camera_make.upper()] = extractor_class
    print(f"Registered camera extractor for {camera_make}")


def get_camera_extractor(
    file_ext: str, 
    exif_data: Dict[str, Any], 
    use_gpu: bool = False, 
    memory_limit: float = 0.75, 
    cpu_cores: Optional[int] = None
) -> Optional[CameraExtractor]:
    """Get the appropriate camera extractor for the given file
    
    Args:
        file_ext: File extension (e.g., '.arw', '.nef')
        exif_data: Basic EXIF data already extracted
        use_gpu: Whether to use GPU acceleration if available
        memory_limit: Memory usage limit as a fraction of total memory
        cpu_cores: Number of CPU cores to use for processing
        
    Returns:
        Camera-specific extractor instance, or None if no suitable extractor is found
    """
    # Get camera make from EXIF data
    camera_make = exif_data.get('camera_make', '').upper()
    
    # First, try to find an extractor based on camera make
    if camera_make in _CAMERA_EXTRACTORS:
        extractor_class = _CAMERA_EXTRACTORS[camera_make]
        extractor = extractor_class(use_gpu=use_gpu, memory_limit=memory_limit, cpu_cores=cpu_cores)
        
        # Check if this extractor can handle the file
        if extractor.can_handle(file_ext, exif_data):
            print(f"Using {extractor_class.__name__} for {camera_make} {file_ext} file")
            return extractor
    
    # If no specific extractor is found, try each registered extractor
    for make, extractor_class in _CAMERA_EXTRACTORS.items():
        extractor = extractor_class(use_gpu=use_gpu, memory_limit=memory_limit, cpu_cores=cpu_cores)
        if extractor.can_handle(file_ext, exif_data):
            print(f"Using {extractor_class.__name__} for {file_ext} file")
            return extractor
    
    # No suitable extractor found
    print(f"No camera-specific extractor found for {camera_make} {file_ext}")
    return None
