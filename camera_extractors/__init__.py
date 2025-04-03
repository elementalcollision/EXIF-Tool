#!/usr/bin/env python3
"""
Camera Extractors Package
Provides modular support for camera-specific EXIF extraction
"""

from .base_extractor import CameraExtractor
from .sony_extractor import SonyExtractor
from .extractor_factory import get_camera_extractor, register_camera_extractor

# Register built-in camera extractors
register_camera_extractor('SONY', SonyExtractor)

# Export public API
__all__ = [
    'CameraExtractor',
    'SonyExtractor',
    'get_camera_extractor',
    'register_camera_extractor'
]
