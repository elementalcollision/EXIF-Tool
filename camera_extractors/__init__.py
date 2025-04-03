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
from .fujifilm_extractor import FujifilmExtractor
from .apple_raw_extractor import AppleRawExtractor
from .panasonic_extractor import PanasonicExtractor
from .olympus_extractor import OlympusExtractor
from .extractor_factory import get_camera_extractor, register_camera_extractor

# Register built-in camera extractors
register_camera_extractor('SONY', SonyExtractor)
register_camera_extractor('NIKON', NikonExtractor)
register_camera_extractor('CANON', CanonExtractor)
register_camera_extractor('DNG', DngExtractor)  # Generic DNG handler
register_camera_extractor('LEICA', DngExtractor)  # Leica-specific handler
register_camera_extractor('FUJI', FujifilmExtractor)  # Fujifilm handler
register_camera_extractor('FUJIFILM', FujifilmExtractor)  # Alternative Fujifilm name
register_camera_extractor('OLYMPUS', OlympusExtractor)  # Olympus handler
register_camera_extractor('APPLE', AppleRawExtractor)  # Apple ProRAW handler
register_camera_extractor('IPHONE', AppleRawExtractor)  # iPhone ProRAW handler
register_camera_extractor('PANASONIC', PanasonicExtractor)  # Panasonic RAW handler
register_camera_extractor('LUMIX', PanasonicExtractor)  # Lumix brand name

# Export public API
__all__ = [
    'CameraExtractor',
    'SonyExtractor',
    'NikonExtractor',
    'CanonExtractor',
    'DngExtractor',
    'FujifilmExtractor',
    'AppleRawExtractor',
    'PanasonicExtractor',
    'OlympusExtractor',
    'get_camera_extractor',
    'register_camera_extractor'
]
