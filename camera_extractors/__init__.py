#!/usr/bin/env python3
"""
Camera Extractors Package
Provides modular support for camera-specific EXIF extraction
"""

from .base_extractor import CameraExtractor
from .sony_extractor import SonyExtractor
from .nikon_extractor import NikonExtractor
from .canon_extractor import CanonExtractor
from .dng_extractor import DngExtractor
from .extractor_factory import get_camera_extractor, register_camera_extractor

# Register built-in camera extractors
register_camera_extractor('SONY', SonyExtractor)
register_camera_extractor('NIKON', NikonExtractor)
register_camera_extractor('CANON', CanonExtractor)
register_camera_extractor('DNG', DngExtractor)  # Generic DNG handler
register_camera_extractor('LEICA', DngExtractor)  # Leica-specific handler

# Export public API
__all__ = [
    'CameraExtractor',
    'SonyExtractor',
    'NikonExtractor',
    'CanonExtractor',
    'DngExtractor',
    'get_camera_extractor',
    'register_camera_extractor'
]
