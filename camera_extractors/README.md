# Camera Extractors Module

This module provides a modular system for adding support for different camera types to the EXIF tool.

## Overview

The camera extractors module allows for easy addition of new camera-specific EXIF extraction capabilities without modifying the core extraction logic. Each camera type (e.g., Sony, Nikon, Canon) has its own extractor class that handles the specific details of that camera's RAW files and metadata.

## Architecture

- `base_extractor.py`: Defines the `CameraExtractor` abstract base class that all camera-specific extractors must implement
- `sony_extractor.py`: Sony-specific implementation for ARW files
- `extractor_factory.py`: Factory pattern for selecting the appropriate extractor based on camera make/model
- `template_extractor.py`: Template for creating new camera-specific extractors

## Adding a New Camera Type

To add support for a new camera type:

1. Create a new file (e.g., `nikon_extractor.py`) based on the template
2. Implement the required methods in your new extractor class
3. Register the extractor in `__init__.py`

### Example: Adding Nikon Support

```python
# nikon_extractor.py
from .base_extractor import CameraExtractor

class NikonExtractor(CameraExtractor):
    def can_handle(self, file_ext, exif_data):
        is_nikon = exif_data.get('camera_make', '').upper().startswith('NIKON')
        is_nef = file_ext.lower() == '.nef'
        return is_nikon and is_nef
    
    # Implement other required methods...

# In __init__.py
from .nikon_extractor import NikonExtractor
register_camera_extractor('NIKON', NikonExtractor)
```

## Required Methods

Each camera extractor must implement:

- `can_handle(file_ext, exif_data)`: Determine if this extractor can handle the given file
- `extract_metadata(image_path, exif_data)`: Extract camera-specific metadata
- `process_raw(image_path, exif_data)`: Process RAW file data
- `get_makernote_tags()`: Get MakerNote tag mapping for this camera type

Optional methods:
- `process_thumbnail(thumb_data, thumb_format)`: Process thumbnail data

## Benefits

- Modular design makes it easy to add support for new camera types
- Camera-specific code is isolated in separate files
- Core extraction logic remains clean and maintainable
- New camera support can be added without modifying existing code
