#!/usr/bin/env python3
"""
Base Camera Extractor
Defines the interface for camera-specific EXIF extractors
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class CameraExtractor(ABC):
    """Base class for camera-specific EXIF extractors"""
    
    def __init__(self, use_gpu=False, memory_limit=0.75, cpu_cores=None):
        """Initialize the camera extractor
        
        Args:
            use_gpu: Whether to use GPU acceleration if available
            memory_limit: Memory usage limit as a fraction of total memory
            cpu_cores: Number of CPU cores to use for processing
        """
        self.use_gpu = use_gpu
        self.memory_limit = memory_limit
        self.cpu_cores = cpu_cores
    
    @abstractmethod
    def can_handle(self, file_ext: str, exif_data: Dict[str, Any]) -> bool:
        """Check if this extractor can handle the given file
        
        Args:
            file_ext: File extension (e.g., '.arw', '.nef')
            exif_data: Basic EXIF data already extracted
            
        Returns:
            True if this extractor can handle the file, False otherwise
        """
        pass
    
    @abstractmethod
    def extract_metadata(self, image_path: str, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract camera-specific metadata from the image
        
        Args:
            image_path: Path to the image file
            exif_data: Basic EXIF data already extracted
            
        Returns:
            Dictionary containing the extracted metadata
        """
        pass
    
    @abstractmethod
    def process_raw(self, image_path: str, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process RAW file data for this camera type
        
        Args:
            image_path: Path to the image file
            exif_data: Basic EXIF data already extracted
            
        Returns:
            Dictionary containing the processed RAW data
        """
        pass
    
    @abstractmethod
    def get_makernote_tags(self) -> Dict[str, str]:
        """Get MakerNote tag mapping for this camera type
        
        Returns:
            Dictionary mapping MakerNote tag names to field names
        """
        pass
    
    def process_thumbnail(self, thumb_data: bytes, thumb_format: str) -> Optional[Dict[str, Any]]:
        """Process thumbnail data (optional, can be overridden by subclasses)
        
        Args:
            thumb_data: Raw thumbnail data
            thumb_format: Format of the thumbnail (e.g., 'jpeg')
            
        Returns:
            Dictionary containing processed thumbnail data, or None
        """
        return None
