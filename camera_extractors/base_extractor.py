#!/usr/bin/env python3
"""
Base Camera Extractor
Defines the interface for camera-specific EXIF extractors
Includes optimization utilities for thread parallelism and memory management
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable, Tuple
from enum import Enum
import os
import time
import logging
import concurrent.futures
import numpy as np
import gc

# Import optimization utilities
from camera_extractors.optimization_utils import (
    ThreadPoolManager, MemoryTracker, parallel_map, 
    chunked_processing, set_memory_limit, performance_monitor,
    safe_array_operation
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('camera_extractor')


class ThumbFormat(Enum):
    """Thumbnail format enum"""
    JPEG = 'jpeg'
    TIFF = 'tiff'


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
        
        # Initialize thread pool manager
        self.thread_pool = ThreadPoolManager(max_workers=cpu_cores)
        
        # Set memory limit for the process
        set_memory_limit(memory_limit)
        
        # Performance metrics
        self.performance_metrics = {}
        
    def __del__(self):
        """Clean up resources when the extractor is deleted"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown_all()
            
    def _time_operation(self, name: str, func: Callable, *args, **kwargs) -> Any:
        """Time an operation and store the result in performance metrics
        
        Args:
            name: Name of the operation
            func: Function to call
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of the function call
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Store performance metric
        self.performance_metrics[name] = end_time - start_time
        
        return result
        
    def parallel_process(self, func: Callable, items: List[Any]) -> List[Any]:
        """Process items in parallel using the thread pool
        
        Args:
            func: Function to apply to each item
            items: List of items to process
            
        Returns:
            List of results
        """
        return parallel_map(func, items, max_workers=self.cpu_cores)
    
    def process_array_in_chunks(self, array: np.ndarray, func: Callable, 
                              chunk_size: int = 1024*1024) -> List[Any]:
        """Process a large array in chunks to limit memory usage
        
        Args:
            array: NumPy array to process
            func: Function to apply to each chunk
            chunk_size: Size of each chunk in elements
            
        Returns:
            List of results from each chunk
        """
        return chunked_processing(array, func, chunk_size, self.memory_limit)
    
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
        
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for this extractor
        
        Returns:
            Dictionary of operation names to execution times in seconds
        """
        return self.performance_metrics
    
    @performance_monitor
    def safe_extract_metadata(self, image_path: str, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Memory-safe wrapper for extract_metadata
        
        Args:
            image_path: Path to the image file
            exif_data: Basic EXIF data already extracted
            
        Returns:
            Dictionary containing the extracted metadata
        """
        with MemoryTracker(self.memory_limit) as tracker:
            return self._time_operation('extract_metadata', 
                                      self.extract_metadata, 
                                      image_path, exif_data)
    
    @performance_monitor
    def safe_process_raw(self, image_path: str, exif_data: Dict[str, Any]) -> Dict[str, Any]:
        """Memory-safe wrapper for process_raw
        
        Args:
            image_path: Path to the image file
            exif_data: Basic EXIF data already extracted
            
        Returns:
            Dictionary containing the processed RAW data
        """
        with MemoryTracker(self.memory_limit) as tracker:
            return self._time_operation('process_raw', 
                                      self.process_raw, 
                                      image_path, exif_data)
