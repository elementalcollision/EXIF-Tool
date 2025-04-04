#!/usr/bin/env python3
"""
EXIF Tool - A tool for processing and visualizing EXIF data from photos
"""

import os
import sys
import csv
import exifread
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import multiprocessing
import psutil
import threading
import concurrent.futures
import platform
import subprocess
import numpy as np
import rawpy
import datetime
import time
import io
import json
from PIL import Image, ImageFile
from exif_db import ExifDatabase
# Disable DecompressionBombWarning
Image.MAX_IMAGE_PIXELS = None  # Disable the DecompressionBombWarning
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Handle truncated images
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# Check if GPU acceleration is available (Metal on Apple Silicon)
GPU_AVAILABLE = False

# Check for Metal support via PyTorch
try:
    import torch
    if torch.backends.mps.is_available():
        print("Metal Performance Shaders (MPS) is available on this Mac")
        GPU_AVAILABLE = True
except ImportError:
    pass

# Check for Metal support via OpenCV
try:
    import cv2
    # Check if OpenCV was built with Metal support
    if hasattr(cv2, 'UMat'):
        # Try to create a small test image with Metal
        try:
            test_img = cv2.UMat(cv2.Mat(np.zeros((10, 10), dtype=np.uint8)))
            # If we get here, Metal is working
            print("OpenCV with Metal acceleration is available")
            GPU_AVAILABLE = True
        except:
            pass
except ImportError:
    pass

# Check for Metal support via CoreML
try:
    import coremltools
    # CoreML uses Metal automatically on Apple Silicon
    print("CoreML with Metal acceleration is available")
    GPU_AVAILABLE = True
except ImportError:
    pass

print(f"GPU acceleration (Metal): {'Available' if GPU_AVAILABLE else 'Not available'}")
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QProgressBar, QTabWidget, QTableWidget, QTableWidgetItem,
                            QComboBox, QMessageBox, QCheckBox, QDialog, QScrollArea)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QIcon
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from tqdm import tqdm
import datetime
import json
from visualization_engine import EnhancedVisualizer

class AppleProRawPanel(QWidget):
    """Panel for displaying Apple ProRAW computational photography features"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout()
        
        # Title and info
        title_label = QLabel("Apple ProRAW Features")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title_label)
        
        info_label = QLabel("Computational photography features detected in Apple ProRAW files")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Feature grid layout
        self.features_layout = QVBoxLayout()
        
        # Create feature indicators
        self.feature_widgets = {}
        self.create_feature_indicators()
        
        # Add features to layout
        feature_widget = QWidget()
        feature_widget.setLayout(self.features_layout)
        layout.addWidget(feature_widget)
        
        # Add a spacer at the bottom
        layout.addStretch(1)
        
        self.setLayout(layout)
    
    def create_feature_indicators(self):
        """Create indicators for each computational photography feature"""
        features = [
            ("apple_hdr", "HDR", "High Dynamic Range processing"),
            ("apple_deep_fusion", "Deep Fusion", "Enhanced detail and reduced noise in medium to low light"),
            ("apple_night_mode", "Night Mode", "Enhanced low-light photography"),
            ("apple_smart_hdr", "Smart HDR", "Intelligent HDR processing"),
            ("apple_photonics_engine", "Photonic Engine", "Deep integration of hardware and software"),
            ("apple_proraw_enabled", "ProRAW", "Apple's RAW format with computational advantages"),
            ("apple_macro_mode", "Macro Mode", "Close-up photography mode"),
            ("apple_photographic_styles", "Photographic Styles", "Custom tone and warmth settings"),
            ("apple_cinematic_mode", "Cinematic Mode", "Depth effects and focus transitions"),
            ("apple_action_mode", "Action Mode", "Stabilization for moving subjects"),
            ("apple_hdr_detected", "HDR (Detected)", "HDR detected through image analysis"),
            ("apple_deep_fusion_detected", "Deep Fusion (Detected)", "Deep Fusion detected through noise analysis")
        ]
        
        for field_name, display_name, description in features:
            # Create a horizontal layout for each feature
            feature_layout = QHBoxLayout()
            
            # Create indicator label (will show ✓ or ✗)
            indicator = QLabel("")
            indicator.setStyleSheet("font-size: 14px; font-weight: bold;")
            feature_layout.addWidget(indicator, 0)
            
            # Create feature name label
            name_label = QLabel(display_name)
            name_label.setToolTip(description)
            feature_layout.addWidget(name_label, 1)
            
            # Add to features layout
            self.features_layout.addLayout(feature_layout)
            
            # Store references to the widgets
            self.feature_widgets[field_name] = (indicator, name_label)
    
    def update_panel(self, exif_data):
        """Update the panel with EXIF data"""
        # Check if this is an Apple ProRAW file
        is_proraw = False
        if exif_data:
            if exif_data.get('file_type') == 'PRORAW' or \
               (exif_data.get('camera_make', '').upper() == 'APPLE' and \
                exif_data.get('file_type', '').upper() == 'DNG'):
                is_proraw = True
        
        # Make the panel visible only for ProRAW files
        self.setVisible(is_proraw)
        
        if not is_proraw:
            return
        
        # Update feature indicators
        for field_name, (indicator, name_label) in self.feature_widgets.items():
            if field_name in exif_data and exif_data[field_name]:
                # Feature is present
                indicator.setText("✓")
                indicator.setStyleSheet("color: green; font-size: 14px; font-weight: bold;")
                name_label.setStyleSheet("font-weight: bold;")
            else:
                # Feature is not present
                indicator.setText("✗")
                indicator.setStyleSheet("color: gray; font-size: 14px;")
                name_label.setStyleSheet("color: gray;")
        
        # Add any additional Apple-specific data
        apple_fields = [key for key in exif_data.keys() if key.startswith('apple_')]
        if apple_fields:
            # Show the number of Apple-specific fields detected
            count_label = QLabel(f"Detected {len(apple_fields)} Apple-specific metadata fields")
            count_label.setStyleSheet("font-style: italic; color: #666;")
            # Check if we already added this label
            if not hasattr(self, 'count_label_added') or not self.count_label_added:
                self.features_layout.addWidget(count_label)
                self.count_label_added = True

class ExifProcessor:
    """Class for processing EXIF data from photos"""
    
    def __init__(self, use_db=True, db_path=None):
        self.supported_extensions = [
            # Common image formats
            '.jpg', '.jpeg', '.tiff', '.tif', '.png', '.heic', '.heif', 
            # Camera RAW formats
            '.nef',  # Nikon
            '.cr2', '.cr3',  # Canon
            '.arw',  # Sony
            '.dng',  # Adobe/Apple/Leica
            '.raf',  # Fujifilm
            '.rw2',  # Panasonic
            '.orf',  # Olympus
            '.pef',  # Pentax
            '.srw',  # Samsung
            '.3fr',  # Hasselblad
            '.mef',  # Mamiya
            '.rwl',  # Leica
            '.mrw'   # Minolta
        ]
        self.exif_data = []
        self.csv_path = None
        self.use_db = use_db
        
        # Initialize database if enabled
        if self.use_db:
            self.db = ExifDatabase(db_path)
            print(f"Using EXIF database at: {self.db.db_path}")
        else:
            self.db = None
        
        # Initialize PIL to handle more formats
        Image.init()
        print(f"Supported PIL formats: {sorted(Image.EXTENSION.keys())}")
    
    def is_image_file(self, filename):
        """Check if the file is a supported image file"""
        ext = os.path.splitext(filename)[1].lower()
        result = ext in self.supported_extensions
        print(f"Checking file: {filename}, extension: {ext}, supported: {result}")
        return result
    
    def process_directory(self, directory_path, callback=None):
        """Process all images in a directory and extract EXIF data"""
        self.exif_data = []
        image_files = []
        
        # Find all image files in the directory and subdirectories
        for root, _, files in os.walk(directory_path):
            for file in files:
                if self.is_image_file(file):
                    image_files.append(os.path.join(root, file))
        
        # Create a collection in the database if using DB
        collection_id = None
        if self.use_db and self.db:
            collection_name = os.path.basename(directory_path)
            collection_id = self.db.create_collection(
                name=collection_name,
                description=f"Images from {directory_path}"
            )
            print(f"Created collection '{collection_name}' with ID: {collection_id}")
        
        # Process each image file
        for i, image_path in enumerate(tqdm(image_files, desc="Processing images")):
            try:
                exif = self.extract_exif(image_path)
                if exif:
                    self.exif_data.append(exif)
                    
                    # Store in database if enabled
                    if self.use_db and self.db and exif:
                        try:
                            # Add to database
                            image_id = self.db.add_image(exif)
                            
                            # Add to collection if we created one
                            if collection_id:
                                self.db.add_image_to_collection(collection_id, image_id)
                        except Exception as db_error:
                            print(f"Database error for {image_path}: {db_error}")
                
                # Update progress if callback is provided
                if callback:
                    progress = (i + 1) / len(image_files) * 100
                    callback(progress, f"Processed {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        # Get database stats if using DB
        if self.use_db and self.db:
            try:
                stats = self.db.get_stats()
                print("\nDatabase Statistics:")
                print(f"Total images: {stats['total_images']}")
                print(f"Images by camera make: {stats['images_by_camera_make']}")
                print(f"Images by file type: {stats['images_by_file_type']}")
            except Exception as stats_error:
                print(f"Error getting database stats: {stats_error}")
        
        print(f"Processed {len(self.exif_data)} images with EXIF data")
        return self.exif_data
    
    def extract_exif(self, image_path):
        """Extract EXIF data from an image file using multiple methods"""
        # Try to use the enhanced extractor first if available
        try:
            from exif_extractor import EnhancedExifExtractor
            print(f"\nUsing enhanced extractor for: {image_path}")
            extractor = EnhancedExifExtractor(use_gpu=GPU_AVAILABLE)
            result = extractor.extract_exif(image_path)
            if result:
                return result
            # If enhanced extractor fails, fall back to built-in method
            print("Enhanced extractor failed, falling back to built-in method")
        except ImportError:
            print("Enhanced extractor not available, using built-in method")
        
        # Fall back to built-in method
        try:
            print(f"\nAttempting to extract EXIF from: {image_path}")
            
            # First check if file exists and is accessible
            if not os.path.isfile(image_path) or not os.access(image_path, os.R_OK):
                print(f"File not accessible: {image_path}")
                return None
                
            # Check file size before processing to avoid very large files
            file_size = os.path.getsize(image_path)
            print(f"File size: {file_size / 1024:.1f} KB")
            
            if file_size > 100 * 1024 * 1024:  # Skip files larger than 100MB
                print(f"Skipping large file ({file_size / (1024*1024):.1f} MB): {image_path}")
                return None
            
            # Initialize variables
            img_width = 0
            img_height = 0
            img_format = None
            creation_date = None
            camera_make = None
            camera_model = None
            exif_fields = {}
            
            # Method 1: Try PIL first (works with most image formats)
            try:
                with Image.open(image_path) as img:
                    img_format = img.format
                    img_width, img_height = img.size
                    print(f"PIL identified format: {img_format}")
                    print(f"Image dimensions: {img_width}x{img_height}")
                    
                    # Try to get EXIF from PIL
                    if hasattr(img, '_getexif') and img._getexif():
                        pil_exif = img._getexif() or {}
                        print(f"PIL found {len(pil_exif)} EXIF tags")
                        
                        # Map PIL EXIF tags (they use numbers as keys)
                        pil_exif_tags = {
                            271: 'camera_make',      # Make
                            272: 'camera_model',     # Model
                            306: 'date_modified',    # DateTime
                            36867: 'date_taken',     # DateTimeOriginal
                            33434: 'exposure_time',  # ExposureTime
                            33437: 'f_number',       # FNumber
                            34855: 'iso',            # ISOSpeedRatings
                            37386: 'focal_length',   # FocalLength
                        }
                        
                        for tag_id, field in pil_exif_tags.items():
                            if tag_id in pil_exif:
                                exif_fields[field] = str(pil_exif[tag_id])
                                if field == 'camera_make':
                                    camera_make = str(pil_exif[tag_id])
                                elif field == 'camera_model':
                                    camera_model = str(pil_exif[tag_id])
                                elif field == 'date_taken':
                                    creation_date = str(pil_exif[tag_id])
            except Exception as pil_error:
                print(f"PIL error: {pil_error}")
            
            # Method 2: Try exifread (better for JPEG and TIFF)
            try:
                with open(image_path, 'rb') as f:
                    tags = exifread.process_file(f, details=False)
                    if tags:
                        print(f"exifread found {len(tags)} tags")
                        
                        # Map exifread tags
                        exifread_mapping = {
                            'Image Make': 'camera_make',
                            'Image Model': 'camera_model',
                            'EXIF DateTimeOriginal': 'date_taken',
                            'EXIF ExposureTime': 'exposure_time',
                            'EXIF FNumber': 'f_number',
                            'EXIF ISOSpeedRatings': 'iso',
                            'EXIF FocalLength': 'focal_length',
                            'EXIF Flash': 'flash',
                            'EXIF ExposureProgram': 'exposure_program',
                            'EXIF MeteringMode': 'metering_mode',
                            'EXIF WhiteBalance': 'white_balance',
                            'GPS GPSLatitude': 'gps_latitude',
                            'GPS GPSLongitude': 'gps_longitude',
                            'GPS GPSAltitude': 'gps_altitude'
                        }
                        
                        for tag, field in exifread_mapping.items():
                            if tag in tags:
                                exif_fields[field] = str(tags[tag])
                                if field == 'camera_make' and not camera_make:
                                    camera_make = str(tags[tag])
                                elif field == 'camera_model' and not camera_model:
                                    camera_model = str(tags[tag])
                                elif field == 'date_taken' and not creation_date:
                                    creation_date = str(tags[tag])
            except Exception as exifread_error:
                print(f"exifread error: {exifread_error}")
            
            # Method 3: Try OpenCV for image properties (works with many formats)
            try:
                import cv2
                img = cv2.imread(image_path)
                if img is not None:
                    h, w = img.shape[:2]
                    if img_width == 0 or img_height == 0:
                        img_width, img_height = w, h
                        print(f"OpenCV identified dimensions: {img_width}x{img_height}")
            except Exception as cv_error:
                print(f"OpenCV error: {cv_error}")
                
            # Method 4: Specialized RAW file processing for all supported RAW formats
            file_ext = os.path.splitext(image_path)[1].lower()
            if file_ext in ['.arw', '.raw', '.nef', '.cr2', '.cr3', '.orf', '.rw2', '.dng', '.raf']:
                try:
                    print(f"Processing RAW file: {file_ext}")
                    
                    # First, try to use camera-specific extractors
                    try:
                        from camera_extractors import get_camera_extractor
                        
                        # Create a basic exif_data dictionary to pass to the extractor
                        basic_exif = {
                            'file_path': image_path,
                            'file_name': os.path.basename(image_path),
                            'file_type': file_ext.lstrip('.').upper(),
                            'camera_make': camera_make,
                            'camera_model': camera_model
                        }
                        
                        # Get appropriate camera extractor with robust error handling
                        try:
                            camera_extractor = get_camera_extractor(
                                file_ext=file_ext,
                                exif_data=basic_exif,
                                use_gpu=GPU_AVAILABLE,
                                memory_limit=0.75,  # Use 75% of available memory
                                cpu_cores=None  # Use default (n-2 on Apple Silicon)
                            )
                            
                            if camera_extractor:
                                print(f"Using camera-specific extractor for {file_ext}")
                                
                                # Extract metadata using the camera-specific extractor with error handling
                                try:
                                    camera_metadata = camera_extractor.extract_metadata(image_path, basic_exif)
                                    if camera_metadata:
                                        for key, value in camera_metadata.items():
                                            exif_fields[key] = value
                                        print(f"Added {len(camera_metadata)} fields from camera-specific extractor")
                                except Exception as metadata_error:
                                    print(f"Error extracting camera metadata: {metadata_error}")
                                    # Continue with processing even if metadata extraction fails
                                
                                # Process RAW data with robust error handling
                                try:
                                    raw_data = camera_extractor.process_raw(image_path, basic_exif)
                                    if raw_data:
                                        for key, value in raw_data.items():
                                            exif_fields[key] = value
                                        print(f"Added {len(raw_data)} fields from RAW processing")
                                except Exception as raw_error:
                                    print(f"Error processing RAW data: {raw_error}")
                                    # Continue with other extraction methods even if RAW processing fails
                        except Exception as extractor_init_error:
                            print(f"Error initializing camera extractor: {extractor_init_error}")
                    except Exception as extractor_error:
                        print(f"Camera extractor error: {extractor_error}")
                    
                    # Fallback to generic rawpy processing
                    with rawpy.imread(image_path) as raw:
                        # Extract basic RAW metadata
                        raw_metadata = {
                            'raw_type': raw.raw_type,
                            'raw_pattern': raw.raw_pattern.tolist() if hasattr(raw, 'raw_pattern') else None,
                            'black_level_per_channel': raw.black_level_per_channel if hasattr(raw, 'black_level_per_channel') else None,
                            'camera_white_level_per_channel': raw.camera_white_level_per_channel if hasattr(raw, 'camera_white_level_per_channel') else None,
                            'color_desc': raw.color_desc.decode('utf-8', errors='ignore') if hasattr(raw, 'color_desc') else None,
                            'num_colors': raw.num_colors if hasattr(raw, 'num_colors') else None,
                            'raw_colors': raw.num_colors if hasattr(raw, 'num_colors') else None,
                            'raw_height': raw.sizes.raw_height if hasattr(raw, 'sizes') else None,
                            'raw_width': raw.sizes.raw_width if hasattr(raw, 'sizes') else None
                        }
                        
                        # Get image dimensions from RAW if not already set
                        if img_width == 0 or img_height == 0:
                            img_width = raw.sizes.width if hasattr(raw, 'sizes') else 0
                            img_height = raw.sizes.height if hasattr(raw, 'sizes') else 0
                            print(f"RAW identified dimensions: {img_width}x{img_height}")
                        
                        # Extract Sony-specific metadata for ARW files
                        if file_ext == '.arw':
                            print("Extracting Sony ARW specific metadata")
                            # Access Sony-specific metadata through rawpy
                            try:
                                # Get raw image data
                                raw_image = raw.raw_image.copy()
                                # Calculate histogram of raw data for exposure analysis
                                histogram, _ = np.histogram(raw_image.flatten(), bins=256)
                                
                                # Sony ARW specific fields
                                sony_metadata = {
                                    'sony_arw_version': 'ARW 2.0' if raw.raw_type == 'ARW2' else 'ARW 1.0',
                                    'sony_raw_histogram_mean': np.mean(histogram),
                                    'sony_raw_histogram_std': np.std(histogram),
                                    'sony_raw_dynamic_range': np.log2(np.max(raw_image) - np.min(raw_image) + 1) if np.max(raw_image) > np.min(raw_image) else 0,
                                    'sony_raw_saturation': np.sum(raw_image >= (raw.white_level - 100)) / raw_image.size if hasattr(raw, 'white_level') else 0,
                                    'sony_raw_black_level': raw.black_level if hasattr(raw, 'black_level') else 0,
                                    'sony_raw_white_level': raw.white_level if hasattr(raw, 'white_level') else 0
                                }
                                
                                # Add Sony-specific metadata to exif_fields
                                for key, value in sony_metadata.items():
                                    exif_fields[key] = value
                                    
                                # If we're using Metal GPU acceleration, process a thumbnail with it
                                if GPU_AVAILABLE:
                                    try:
                                        # Use Metal to process the RAW thumbnail
                                        import torch
                                        if torch.backends.mps.is_available():
                                            # Get thumbnail from RAW
                                            thumb = raw.extract_thumb()
                                            if thumb.format == 'jpeg':
                                                # Convert thumbnail to tensor and process with Metal
                                                with Image.open(io.BytesIO(thumb.data)) as pil_thumb:
                                                    # Convert PIL image to tensor and move to MPS
                                                    device = torch.device("mps")
                                                    thumb_tensor = torch.from_numpy(np.array(pil_thumb)).to(device)
                                                    # Apply some basic processing
                                                    thumb_tensor = thumb_tensor / 255.0  # Normalize
                                                    # Simple brightness adjustment
                                                    thumb_tensor = torch.clamp(thumb_tensor * 1.2, 0, 1)
                                                    print("Processed RAW thumbnail with Metal acceleration")
                                    except Exception as metal_error:
                                        print(f"Metal processing error: {metal_error}")
                                        
                            except Exception as sony_error:
                                print(f"Sony ARW specific extraction error: {sony_error}")
                        
                        # Add general RAW metadata to exif_fields
                        for key, value in raw_metadata.items():
                            if value is not None:
                                exif_fields[key] = value
                                
                except Exception as raw_error:
                    print(f"RAW processing error: {raw_error}")
            
            # Method 5: Use file metadata as fallback
            try:
                file_stat = os.stat(image_path)
                file_created = datetime.datetime.fromtimestamp(file_stat.st_ctime)
                file_modified = datetime.datetime.fromtimestamp(file_stat.st_mtime)
                
                # Use file creation date if no EXIF date was found
                if not creation_date:
                    creation_date = file_created.strftime('%Y:%m:%d %H:%M:%S')
                    exif_fields['date_taken'] = creation_date
                    print(f"Using file creation date: {creation_date}")
            except Exception as stat_error:
                print(f"File stat error: {stat_error}")
            
            # Always create a record with whatever data we have
            print(f"Extracted {len(exif_fields)} EXIF fields")
            
            # Create the EXIF record with all available data
            exif = {
                'file_path': image_path,
                'file_name': os.path.basename(image_path),
                'file_size': file_size,
                'file_type': img_format or os.path.splitext(image_path)[1].lstrip('.').upper(),
                'width': img_width,
                'height': img_height,
                'date_processed': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Add all extracted EXIF fields
            exif.update(exif_fields)
            
            # Add camera info to the record name if available
            if camera_make and camera_model:
                exif['camera'] = f"{camera_make} {camera_model}"
            elif camera_make:
                exif['camera'] = camera_make
            elif camera_model:
                exif['camera'] = camera_model
            
            return exif
        except Exception as e:
            print(f"Error extracting EXIF from {image_path}: {e}")
            # Even if we encounter an error, try to return basic file info
            try:
                return {
                    'file_path': image_path,
                    'file_name': os.path.basename(image_path),
                    'file_size': os.path.getsize(image_path),
                    'date_processed': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'error': str(e)
                }
            except:
                return None
    
    def save_to_csv(self, output_path, collection_id=None, include_normalized=True):
        """Save EXIF data to a CSV file
        
        Args:
            output_path: Path to save the CSV file
            collection_id: Optional collection ID to filter images
            include_normalized: Whether to include normalized EXIF fields
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # If using database, export directly from database
            if self.use_db and self.db:
                try:
                    self.db.export_to_csv(output_path, collection_id, include_normalized=include_normalized)
                    print(f"Exported EXIF data from database to {output_path}")
                    if include_normalized:
                        print("Normalized EXIF fields included in export")
                    self.csv_path = output_path
                    return True
                except Exception as db_error:
                    print(f"Database export error: {db_error}")
                    # Fall back to in-memory data if database export fails
            
            # Export from in-memory data if not using database or if database export failed
            if not self.exif_data:
                return False
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Get all unique keys from all dictionaries
            all_keys = set()
            for exif in self.exif_data:
                all_keys.update(exif.keys())
            
            # Write to CSV
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=sorted(all_keys))
                writer.writeheader()
                writer.writerows(self.exif_data)
            
            self.csv_path = output_path
            return True
        except Exception as e:
            print(f"Error saving CSV: {e}")
            return False
    
    def load_from_csv(self, csv_path):
        """Load EXIF data from a CSV file
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load data from CSV into memory
            self.exif_data = pd.read_csv(csv_path).to_dict('records')
            self.csv_path = csv_path
            
            # Optionally add to database if using it
            if self.use_db and self.db:
                print(f"Importing {len(self.exif_data)} records to database...")
                # Create a collection for this import
                collection_name = os.path.basename(csv_path).replace('.csv', '')
                collection_id = self.db.create_collection(
                    name=f"Import: {collection_name}",
                    description=f"Imported from {csv_path} on {datetime.datetime.now()}"
                )
                
                # Add each record to the database
                for i, record in enumerate(tqdm(self.exif_data, desc="Importing to database")):
                    try:
                        # Add to database
                        image_id = self.db.add_image(record)
                        # Add to collection
                        self.db.add_image_to_collection(collection_id, image_id)
                    except Exception as db_error:
                        print(f"Database import error for record {i}: {db_error}")
                
                print(f"Imported data to database collection '{collection_name}'")
            
            return True
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False
    
    def get_data_frame(self):
        """Convert EXIF data to a pandas DataFrame"""
        return pd.DataFrame(self.exif_data)


class ExifAnalyzer:
    """Class for analyzing and visualizing EXIF data"""
    
    def __init__(self, exif_data=None):
        self.exif_data = exif_data or []
        self.df = pd.DataFrame(self.exif_data)
    
    def set_data(self, exif_data):
        """Set the EXIF data to analyze"""
        self.exif_data = exif_data
        self.df = pd.DataFrame(exif_data)
    
    def load_csv(self, csv_path):
        """Load EXIF data from a CSV file"""
        try:
            # Read CSV with explicit date parsing
            self.df = pd.read_csv(csv_path, parse_dates=['date_taken', 'date_processed'], 
                                 infer_datetime_format=True, errors='coerce')
            
            # Convert to records
            self.exif_data = self.df.to_dict('records')
            return True
        except Exception as e:
            print(f"Error loading CSV for analysis: {e}")
            return False
    
    def get_summary_stats(self):
        """Get summary statistics of the EXIF data"""
        if self.df.empty:
            return {}
        
        stats = {
            'total_images': len(self.df),
            'unique_cameras': self.df['camera_model'].nunique() if 'camera_model' in self.df.columns else 0,
            'date_range': None
        }
        
        # Get date range if date_taken exists
        if 'date_taken' in self.df.columns and not self.df['date_taken'].isna().all():
            try:
                # Common EXIF date formats
                date_formats = [
                    '%Y:%m:%d %H:%M:%S',  # Standard EXIF format
                    '%Y-%m-%d %H:%M:%S',  # ISO format
                    '%Y-%m-%dT%H:%M:%S',  # ISO format with T separator
                    '%Y-%m-%d',           # Date only
                    '%Y:%m:%d'            # EXIF date only
                ]
                
                # Try to parse dates with explicit formats first
                parsed = False
                for date_format in date_formats:
                    try:
                        self.df['date_taken_dt'] = pd.to_datetime(self.df['date_taken'], 
                                                                 format=date_format, 
                                                                 errors='coerce')
                        if not self.df['date_taken_dt'].isna().all():
                            parsed = True
                            break
                    except:
                        continue
                
                # Fall back to automatic parsing if needed
                if not parsed:
                    self.df['date_taken_dt'] = pd.to_datetime(self.df['date_taken'], errors='coerce')
                
                # Get min and max dates
                min_date = self.df['date_taken_dt'].min()
                max_date = self.df['date_taken_dt'].max()
                if pd.notna(min_date) and pd.notna(max_date):
                    stats['date_range'] = (min_date.strftime('%Y-%m-%d'), max_date.strftime('%Y-%m-%d'))
            except Exception as e:
                print(f"Error parsing dates: {e}")
        
        return stats
    
    def plot_camera_distribution(self, figure=None):
        """Plot distribution of camera models"""
        if self.df.empty or 'camera_model' not in self.df.columns:
            return None
        
        if figure is None:
            fig = Figure(figsize=(10, 6))
        else:
            fig = figure
            fig.clear()
        
        ax = fig.add_subplot(111)
        
        # Count camera models
        camera_counts = self.df['camera_model'].value_counts().head(10)
        
        # Plot horizontal bar chart
        bars = ax.barh(camera_counts.index, camera_counts.values, color=sns.color_palette("viridis", len(camera_counts)))
        
        # Add count labels
        for i, bar in enumerate(bars):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                   str(camera_counts.values[i]), 
                   va='center')
        
        ax.set_title('Top 10 Camera Models')
        ax.set_xlabel('Number of Images')
        fig.tight_layout()
        
        return fig
    
    def plot_focal_length_distribution(self, figure=None):
        """Plot distribution of focal lengths"""
        if self.df.empty or 'focal_length' not in self.df.columns:
            return None
        
        # Extract numeric focal length values
        self.df['focal_length_num'] = self.df['focal_length'].str.extract(r'(\d+)').astype(float)
        
        if figure is None:
            fig = Figure(figsize=(10, 6))
        else:
            fig = figure
            fig.clear()
        
        ax = fig.add_subplot(111)
        
        # Create histogram
        ax.hist(self.df['focal_length_num'].dropna(), bins=20, alpha=0.7, color='skyblue')
        
        ax.set_title('Focal Length Distribution')
        ax.set_xlabel('Focal Length (mm)')
        ax.set_ylabel('Number of Images')
        fig.tight_layout()
        
        return fig
    
    def plot_aperture_distribution(self, figure=None):
        """Plot distribution of aperture values (f-numbers)"""
        if self.df.empty or 'f_number' not in self.df.columns:
            return None
        
        # Extract numeric f-number values
        self.df['f_number_num'] = self.df['f_number'].str.extract(r'f/(\d+\.?\d*)').astype(float)
        
        if figure is None:
            fig = Figure(figsize=(10, 6))
        else:
            fig = figure
            fig.clear()
        
        ax = fig.add_subplot(111)
        
        # Create histogram
        ax.hist(self.df['f_number_num'].dropna(), bins=15, alpha=0.7, color='lightgreen')
        
        ax.set_title('Aperture (f-number) Distribution')
        ax.set_xlabel('f-number')
        ax.set_ylabel('Number of Images')
        fig.tight_layout()
        
        return fig
    
    def plot_iso_distribution(self, figure=None):
        """Plot distribution of ISO values"""
        if self.df.empty or 'iso' not in self.df.columns:
            return None
        
        # Extract numeric ISO values
        self.df['iso_num'] = pd.to_numeric(self.df['iso'], errors='coerce')
        
        if figure is None:
            fig = Figure(figsize=(10, 6))
        else:
            fig = figure
            fig.clear()
        
        ax = fig.add_subplot(111)
        
        # Create histogram with logarithmic x-axis
        ax.hist(self.df['iso_num'].dropna(), bins=20, alpha=0.7, color='salmon')
        ax.set_xscale('log')
        
        ax.set_title('ISO Distribution')
        ax.set_xlabel('ISO (log scale)')
        ax.set_ylabel('Number of Images')
        fig.tight_layout()
        
        return fig
    
    def plot_time_of_day(self, figure=None):
        """Plot distribution of photos by time of day"""
        if self.df.empty or 'date_taken' not in self.df.columns:
            return None
        
        try:
            # Common EXIF date formats
            date_formats = [
                '%Y:%m:%d %H:%M:%S',  # Standard EXIF format
                '%Y-%m-%d %H:%M:%S',  # ISO format
                '%Y-%m-%dT%H:%M:%S',  # ISO format with T separator
            ]
            
            # Try to parse dates with explicit formats first
            parsed = False
            for date_format in date_formats:
                try:
                    self.df['date_taken_dt'] = pd.to_datetime(self.df['date_taken'], 
                                                             format=date_format, 
                                                             errors='coerce')
                    if not self.df['date_taken_dt'].isna().all():
                        parsed = True
                        break
                except:
                    continue
            
            # Fall back to automatic parsing if needed
            if not parsed:
                self.df['date_taken_dt'] = pd.to_datetime(self.df['date_taken'], errors='coerce')
            
            # Extract hour from datetime
            self.df['hour'] = self.df['date_taken_dt'].dt.hour
            
            # Skip if no valid hours were extracted
            if self.df['hour'].isna().all():
                return None
            
            if figure is None:
                fig = Figure(figsize=(10, 6))
            else:
                fig = figure
                fig.clear()
            
            ax = fig.add_subplot(111)
            
            # Create histogram of hours
            hour_counts = self.df['hour'].value_counts().sort_index()
            ax.bar(hour_counts.index, hour_counts.values, color='purple', alpha=0.7)
            
            ax.set_title('Photos by Time of Day')
            ax.set_xlabel('Hour of Day (24h)')
            ax.set_ylabel('Number of Images')
            ax.set_xticks(range(0, 24, 2))
            fig.tight_layout()
            
            return fig
        except Exception as e:
            print(f"Error creating time of day plot: {e}")
            return None


class ResourceMonitor:
    """Monitor and manage system resources optimized for Apple Silicon"""
    
    def __init__(self, memory_limit=0.75):
        self.memory_limit = memory_limit
        self.process = psutil.Process(os.getpid())
        
        # Check if running on Apple Silicon
        self.is_apple_silicon = self._check_apple_silicon()
        if self.is_apple_silicon:
            print("Running on Apple Silicon - optimizing resource usage")
    
    def _check_apple_silicon(self):
        """Check if running on Apple Silicon"""
        try:
            # Check platform for macOS
            if platform.system() != "Darwin":
                return False
                
            # Check processor info for Apple Silicon
            cmd = "sysctl -n machdep.cpu.brand_string"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return "Apple" in result.stdout
        except:
            # Default to False if we can't determine
            return False
    
    def check_memory_usage(self):
        """Check if memory usage is below the limit"""
        if self.memory_limit is None:
            return True
        
        # Get current memory usage
        current_memory = self.process.memory_info().rss
        
        if self.is_apple_silicon:
            # On Apple Silicon, account for unified memory architecture
            # where RAM is shared between CPU and GPU
            total_memory = psutil.virtual_memory().total
            # Reserve a bit more memory for the Metal GPU on Apple Silicon
            effective_total = total_memory * 0.9  # Reserve 10% for Metal GPU
            memory_usage_ratio = current_memory / effective_total
        else:
            total_memory = psutil.virtual_memory().total
            memory_usage_ratio = current_memory / total_memory
        
        return memory_usage_ratio < self.memory_limit
    
    def get_apple_silicon_core_usage(self):
        """Get efficiency and performance core usage on Apple Silicon"""
        if not self.is_apple_silicon:
            return None
            
        try:
            # Get per-core CPU usage
            per_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
            core_count = len(per_cpu)
            
            # Apple Silicon typically has efficiency cores first
            # M1: 4 efficiency, 4 performance cores
            # M2: 4 efficiency, 4-8 performance cores
            # M3: 4 efficiency, 4-8 performance cores
            # This is a simplification - actual mapping may vary
            e_core_count = min(4, core_count // 3)  # Estimate efficiency cores as ~1/3 of total
            
            return {
                'efficiency_cores': sum(per_cpu[:e_core_count]) / e_core_count if e_core_count > 0 else 0,
                'performance_cores': sum(per_cpu[e_core_count:]) / (core_count - e_core_count) if (core_count - e_core_count) > 0 else 0,
                'overall': sum(per_cpu) / core_count if core_count > 0 else 0
            }
        except:
            return None
    
    def wait_for_memory(self, interval=0.5, max_wait=30):
        """Wait until memory usage is below the limit"""
        if self.memory_limit is None:
            return True
        
        start_time = time.time()
        while not self.check_memory_usage():
            # If we've waited too long, return False
            if time.time() - start_time > max_wait:
                return False
            
            # On Apple Silicon, wait a bit longer to allow Metal to release memory
            if self.is_apple_silicon:
                time.sleep(interval * 1.5)  # 50% longer wait for Metal memory management
            else:
                time.sleep(interval)
        
        return True


class ProcessingThread(QThread):
    """Thread for processing images in the background"""
    progress_update = pyqtSignal(float, str)
    processing_complete = pyqtSignal(list)
    
    def __init__(self, directory_path, max_image_size=None, recursive=True, skip_no_exif=True,
                 cpu_cores=None, memory_limit=0.75, use_gpu=False, gpu_image_processing=False):
        super().__init__()
        self.directory_path = directory_path
        self.max_image_size = max_image_size
        self.recursive = recursive
        self.skip_no_exif = skip_no_exif
        
        # Resource settings
        self.cpu_cores = cpu_cores if cpu_cores is not None else max(1, multiprocessing.cpu_count() - 2)
        self.memory_limit = memory_limit
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.gpu_image_processing = gpu_image_processing and self.use_gpu
        
        self.processor = ExifProcessor()
        self.resource_monitor = ResourceMonitor(memory_limit)
    
    def process_image(self, image_path):
        """Process a single image file"""
        try:
            # Check file size if a limit is set
            if self.max_image_size is not None:
                file_size = os.path.getsize(image_path)
                if file_size > self.max_image_size:
                    return None, f"Skipping large file ({file_size / (1024*1024):.1f} MB): {os.path.basename(image_path)}"
            
            # Wait for memory to be available if needed
            if not self.resource_monitor.wait_for_memory():
                return None, f"Memory limit reached, skipping: {os.path.basename(image_path)}"
            
            # Extract EXIF data - use Metal GPU acceleration if available and enabled
            if self.use_gpu and self.gpu_image_processing:
                # Use Metal-accelerated image processing if available
                try:
                    # Use Metal acceleration via OpenCV UMat
                    try:
                        import cv2
                        if hasattr(cv2, 'UMat'):
                            # Read image with Metal acceleration
                            img = cv2.imread(image_path)
                            if img is not None:
                                # Convert to UMat for GPU processing
                                gpu_img = cv2.UMat(img)
                                # Process on GPU (e.g., resize, convert color space)
                                gpu_img = cv2.resize(gpu_img, (gpu_img.width, gpu_img.height))
                                # Download back to CPU for EXIF extraction
                                _ = gpu_img.get()
                                print("Used Metal acceleration via OpenCV")
                    except Exception as cv_error:
                        print(f"OpenCV Metal acceleration failed: {cv_error}")
                    
                    # Try PyTorch MPS (Metal Performance Shaders)
                    try:
                        import torch
                        if torch.backends.mps.is_available():
                            # Create MPS device
                            device = torch.device("mps")
                            # Load image data using PIL and convert to tensor
                            with Image.open(image_path) as pil_img:
                                # Convert PIL image to tensor and move to MPS
                                img_tensor = torch.from_numpy(np.array(pil_img)).to(device)
                                # Perform some GPU operations (e.g., normalization)
                                img_tensor = img_tensor / 255.0
                                # Move back to CPU
                                _ = img_tensor.cpu().numpy()
                                print("Used Metal acceleration via PyTorch MPS")
                    except Exception as torch_error:
                        print(f"PyTorch MPS acceleration failed: {torch_error}")
                    
                    # Extract EXIF data after GPU preprocessing
                    exif = self.processor.extract_exif(image_path)
                except Exception as gpu_error:
                    # Fall back to CPU if GPU processing fails
                    print(f"Metal GPU processing failed, falling back to CPU: {gpu_error}")
                    exif = self.processor.extract_exif(image_path)
            else:
                # Use standard CPU-based processing
                exif = self.processor.extract_exif(image_path)
            
            # With our improved extract_exif method, we should always get data
            # But just in case, create a minimal record if exif is None
            if exif is None:
                if self.skip_no_exif:
                    return None, f"Skipping file with no EXIF data: {os.path.basename(image_path)}"
                else:
                    # Create minimal record with file info
                    exif = {
                        'file_path': image_path,
                        'file_name': os.path.basename(image_path),
                        'file_size': os.path.getsize(image_path),
                        'date_processed': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'note': 'No EXIF data found'
                    }
            
            # Add file extension as a field if not already present
            if 'file_type' not in exif:
                exif['file_type'] = os.path.splitext(image_path)[1].lstrip('.').upper()
                
            return exif, f"Processed: {os.path.basename(image_path)}"
        except Exception as e:
            # Even on error, create a minimal record
            try:
                minimal_exif = {
                    'file_path': image_path,
                    'file_name': os.path.basename(image_path),
                    'file_size': os.path.getsize(image_path),
                    'date_processed': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'error': str(e)
                }
                return minimal_exif, f"Processed with errors: {os.path.basename(image_path)}"
            except:
                return None, f"Error processing {os.path.basename(image_path)}: {str(e)}"
    
    def run(self):
        """Run the processing thread"""
        # Update the processor with our settings
        exif_data = []
        image_files = []
        
        # Find all image files in the directory
        self.progress_update.emit(0, "Finding image files...")
        print(f"\n\nSearching for images in: {self.directory_path}")
        print(f"Recursive mode: {self.recursive}")
        print(f"Supported extensions: {self.processor.supported_extensions}")
        
        # Count all files to check if we're finding any files at all
        total_files = 0
        
        if self.recursive:
            for root, dirs, files in os.walk(self.directory_path):
                print(f"Checking directory: {root} ({len(files)} files)")
                total_files += len(files)
                for file in files:
                    file_path = os.path.join(root, file)
                    # Check file extension directly to avoid case sensitivity issues
                    _, ext = os.path.splitext(file)
                    if ext.lower() in self.processor.supported_extensions:
                        print(f"Found image: {file_path}")
                        image_files.append(file_path)
        else:
            # Non-recursive mode - only process files in the top directory
            files = os.listdir(self.directory_path)
            print(f"Non-recursive mode, found {len(files)} files in directory")
            total_files = len(files)
            for file in files:
                file_path = os.path.join(self.directory_path, file)
                if os.path.isfile(file_path):
                    # Check file extension directly to avoid case sensitivity issues
                    _, ext = os.path.splitext(file)
                    if ext.lower() in self.processor.supported_extensions:
                        print(f"Found image: {file_path}")
                        image_files.append(file_path)
        
        print(f"Total files checked: {total_files}")
        print(f"Found {len(image_files)} image files with supported extensions")
        
        if not image_files:
            self.progress_update.emit(100, f"No image files found (checked {total_files} files)")
            self.processing_complete.emit([])
            return
        
        self.progress_update.emit(5, f"Found {len(image_files)} image files. Processing...")
        
        # Process images using a thread pool with controlled number of workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.cpu_cores) as executor:
            # Submit all tasks
            future_to_image = {executor.submit(self.process_image, image_path): image_path 
                              for image_path in image_files}
            
            # Process results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(future_to_image)):
                image_path = future_to_image[future]
                try:
                    exif, message = future.result()
                    if exif:
                        exif_data.append(exif)
                    
                    # Update progress
                    progress = 5 + (i + 1) / len(image_files) * 95  # Scale from 5% to 100%
                    self.progress_update.emit(progress, message)
                except Exception as e:
                    self.progress_update.emit(
                        5 + (i + 1) / len(image_files) * 95,
                        f"Error processing {os.path.basename(image_path)}: {str(e)}"
                    )
        
        # Store the data in the processor and emit completion signal
        self.processor.exif_data = exif_data
        self.processing_complete.emit(exif_data)


class ConfigDialog(QDialog):
    """Configuration dialog for EXIF Tool settings"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("EXIF Tool Settings")
        self.setGeometry(200, 200, 500, 500)
        self.setModal(True)  # Make dialog modal
        
        # Get system information
        self.total_cores = multiprocessing.cpu_count()
        self.total_memory = psutil.virtual_memory().total
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        
        # Create tab widget for settings categories
        tabs = QTabWidget()
        
        # Basic settings tab
        basic_tab = QWidget()
        basic_layout = QVBoxLayout()
        
        # Image size limit
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Maximum image size (MB):"))
        self.size_limit = QComboBox()
        self.size_limit.addItems(["10", "50", "100", "200", "500", "No limit"])
        self.size_limit.setCurrentText("No limit")
        size_layout.addWidget(self.size_limit)
        basic_layout.addLayout(size_layout)
        
        # Recursive processing
        self.recursive_check = QCheckBox("Process subdirectories recursively")
        self.recursive_check.setChecked(True)
        basic_layout.addWidget(self.recursive_check)
        
        # Skip images without EXIF data
        self.skip_no_exif = QCheckBox("Skip images without EXIF data")
        self.skip_no_exif.setChecked(True)
        basic_layout.addWidget(self.skip_no_exif)
        
        basic_tab.setLayout(basic_layout)
        tabs.addTab(basic_tab, "Basic")
        
        # Resource management tab
        resource_tab = QWidget()
        resource_layout = QVBoxLayout()
        
        # CPU cores section
        resource_layout.addWidget(QLabel("<b>CPU Settings</b>"))
        
        # CPU cores
        cores_layout = QHBoxLayout()
        cores_layout.addWidget(QLabel(f"CPU Cores to use (Total available: {self.total_cores}):"))
        self.cpu_cores = QComboBox()
        
        # Calculate n-2 cores as default
        default_cores = max(1, self.total_cores - 2)
        core_options = [str(i) for i in range(1, self.total_cores + 1)]
        core_options.append("Auto (n-2)")
        
        self.cpu_cores.addItems(core_options)
        self.cpu_cores.setCurrentText("Auto (n-2)")
        cores_layout.addWidget(self.cpu_cores)
        resource_layout.addLayout(cores_layout)
        
        # Memory limit
        memory_layout = QHBoxLayout()
        memory_layout.addWidget(QLabel("Memory usage limit:"))
        self.memory_limit = QComboBox()
        self.memory_limit.addItems(["25%", "50%", "75%", "90%", "No limit"])
        self.memory_limit.setCurrentText("75%")
        memory_layout.addWidget(self.memory_limit)
        resource_layout.addLayout(memory_layout)
        
        # Memory info
        total_gb = round(self.total_memory / (1024**3), 1)
        memory_info = QLabel(f"Total system memory: {total_gb} GB")
        resource_layout.addWidget(memory_info)
        
        resource_layout.addSpacing(15)
        
        # GPU acceleration section (Metal for Apple Silicon)
        resource_layout.addWidget(QLabel("<b>Apple Silicon GPU Acceleration (Metal)</b>"))
        
        # Metal GPU acceleration checkbox
        self.use_gpu = QCheckBox("Enable Metal GPU acceleration")
        self.use_gpu.setChecked(True)
        if not GPU_AVAILABLE:
            self.use_gpu.setEnabled(False)
            self.use_gpu.setChecked(False)
            resource_layout.addWidget(QLabel("Metal GPU acceleration not available on this Mac"))
        else:
            resource_layout.addWidget(QLabel("Metal Performance Shaders detected on this Apple Silicon Mac"))
        resource_layout.addWidget(self.use_gpu)
        
        # GPU operations
        gpu_ops_layout = QVBoxLayout()
        gpu_ops_layout.addWidget(QLabel("Metal-accelerated operations:"))
        
        self.gpu_image_processing = QCheckBox("Image processing (Metal accelerated)")
        self.gpu_image_processing.setChecked(True)
        self.gpu_image_processing.setEnabled(GPU_AVAILABLE)
        gpu_ops_layout.addWidget(self.gpu_image_processing)
        
        self.gpu_data_analysis = QCheckBox("Data analysis (Metal accelerated)")
        self.gpu_data_analysis.setChecked(True)
        self.gpu_data_analysis.setEnabled(GPU_AVAILABLE)
        gpu_ops_layout.addWidget(self.gpu_data_analysis)
        
        # Add Metal performance note
        if GPU_AVAILABLE:
            metal_note = QLabel("<i>Note: Metal acceleration is optimized for Apple Silicon and may significantly improve performance.</i>")
            metal_note.setWordWrap(True)
            gpu_ops_layout.addWidget(metal_note)
        
        resource_layout.addLayout(gpu_ops_layout)
        
        resource_tab.setLayout(resource_layout)
        tabs.addTab(resource_tab, "Resources")
        
        # Add tabs to main layout
        layout.addWidget(tabs)
        
        # Button layout
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self.save_settings)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.close)
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def save_settings(self):
        """Save settings and close dialog"""
        # Basic settings
        # Get max image size
        size_text = self.size_limit.currentText()
        if size_text == "No limit":
            self.parent.max_image_size = None
        else:
            self.parent.max_image_size = int(size_text) * 1024 * 1024  # Convert MB to bytes
        
        # Get other basic settings
        self.parent.recursive = self.recursive_check.isChecked()
        self.parent.skip_no_exif = self.skip_no_exif.isChecked()
        
        # Resource settings
        # CPU cores
        cores_text = self.cpu_cores.currentText()
        if cores_text == "Auto (n-2)":
            self.parent.cpu_cores = max(1, self.total_cores - 2)
        else:
            self.parent.cpu_cores = int(cores_text)
        
        # Memory limit
        memory_text = self.memory_limit.currentText()
        if memory_text == "No limit":
            self.parent.memory_limit = None
        else:
            # Convert percentage to fraction
            percentage = int(memory_text.strip('%'))
            self.parent.memory_limit = percentage / 100.0
        
        # GPU settings
        self.parent.use_gpu = self.use_gpu.isChecked() and GPU_AVAILABLE
        self.parent.gpu_image_processing = self.gpu_image_processing.isChecked() and self.parent.use_gpu
        self.parent.gpu_data_analysis = self.gpu_data_analysis.isChecked() and self.parent.use_gpu
        
        self.close()


class ExifToolGUI(QMainWindow):
    """Main GUI for the EXIF Tool"""
    
    def __init__(self):
        super().__init__()
        self.processor = ExifProcessor()
        self.analyzer = ExifAnalyzer()
        
        # Default basic settings
        self.max_image_size = None  # No limit
        self.recursive = True
        self.skip_no_exif = True
        
        # Default resource settings
        self.total_cores = multiprocessing.cpu_count()
        self.cpu_cores = max(1, self.total_cores - 2)  # n-2 cores by default
        self.memory_limit = 0.75  # 75% of available memory
        
        # GPU settings
        self.use_gpu = GPU_AVAILABLE
        self.gpu_image_processing = GPU_AVAILABLE
        self.gpu_data_analysis = GPU_AVAILABLE
        
        # Set up event handling for application close
        app = QApplication.instance()
        app.aboutToQuit.connect(self.save_preferences)
        
        # Define visualizable fields (fields that are useful for visualization)
        self.visualizable_fields = [
            'file_name', 'file_path', 'file_type', 'file_size',
            'camera_make', 'camera_model', 'lens_model',
            'focal_length', 'f_number', 'iso', 'exposure_time', 'shutter_speed',
            'exposure_program', 'metering_mode', 'white_balance',
            'date_time', 'gps_latitude', 'gps_longitude', 'altitude',
            'width', 'height', 'orientation',
            'flash', 'scene_type', 'scene_capture_type'
        ]
        
        # User preferences file path
        self.config_dir = os.path.join(os.path.expanduser("~"), ".exif_tool")
        self.preferences_file = os.path.join(self.config_dir, "preferences.json")
        
        # Create config directory if it doesn't exist
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir, exist_ok=True)
        
        # Load saved preferences or use defaults
        self.load_preferences()
        
        # Initialize enhanced visualizer with GPU acceleration if available
        self.visualizer = EnhancedVisualizer(
            use_gpu=self.use_gpu,
            memory_limit=self.memory_limit,
            cpu_cores=self.cpu_cores
        )
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("EXIF Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Top controls
        controls_layout = QHBoxLayout()
        
        self.select_dir_btn = QPushButton("Select Directory")
        self.select_dir_btn.clicked.connect(self.select_directory)
        controls_layout.addWidget(self.select_dir_btn)
        
        self.load_csv_btn = QPushButton("Load CSV")
        self.load_csv_btn.clicked.connect(self.load_csv)
        controls_layout.addWidget(self.load_csv_btn)
        
        self.export_csv_btn = QPushButton("Export to CSV")
        self.export_csv_btn.clicked.connect(self.export_to_csv)
        self.export_csv_btn.setEnabled(False)
        controls_layout.addWidget(self.export_csv_btn)
        
        # Add field selector button
        self.field_selector_btn = QPushButton("Select Fields")
        self.field_selector_btn.clicked.connect(self.show_field_selector)
        self.field_selector_btn.setEnabled(False)
        controls_layout.addWidget(self.field_selector_btn)
        
        # Add settings button
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.clicked.connect(self.show_settings)
        controls_layout.addWidget(self.settings_btn)
        
        main_layout.addLayout(controls_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)
        
        # Tab widget
        self.tabs = QTabWidget()
        
        # Data tab
        self.data_tab = QWidget()
        data_layout = QVBoxLayout()
        
        self.data_table = QTableWidget()
        # Enable sorting
        self.data_table.setSortingEnabled(True)
        # Connect header click to custom sort function
        self.data_table.horizontalHeader().sectionClicked.connect(self.on_header_clicked)
        # Track current sort order for each column
        self.sort_orders = {}
        
        data_layout.addWidget(self.data_table)
        
        self.data_tab.setLayout(data_layout)
        self.tabs.addTab(self.data_tab, "Data")
        
        # Summary tab
        self.summary_tab = QWidget()
        summary_layout = QVBoxLayout()
        
        self.summary_label = QLabel("No data loaded")
        summary_layout.addWidget(self.summary_label)
        
        self.summary_tab.setLayout(summary_layout)
        self.tabs.addTab(self.summary_tab, "Summary")
        
        # Visualization tab
        self.viz_tab = QWidget()
        viz_layout = QVBoxLayout()
        
        # Visualization type selector
        viz_controls = QHBoxLayout()
        self.viz_type_combo = QComboBox()
        self.viz_type_combo.addItems([
            "Camera Distribution", 
            "Focal Length Distribution",
            "Aperture Distribution",
            "ISO Distribution",
            "Time of Day Distribution",
            "Location Map",
            "Aperture vs Focal Length",
            "ISO vs Time of Day"
        ])
        self.viz_type_combo.currentIndexChanged.connect(self.update_visualization)
        viz_controls.addWidget(QLabel("Visualization:"))
        viz_controls.addWidget(self.viz_type_combo)
        viz_layout.addLayout(viz_controls)
        
        # Additional visualization controls
        viz_options = QHBoxLayout()
        
        # GPU acceleration toggle
        self.gpu_viz_checkbox = QCheckBox("Use GPU")
        self.gpu_viz_checkbox.setChecked(self.use_gpu)
        self.gpu_viz_checkbox.setToolTip("Toggle GPU acceleration for visualizations")
        self.gpu_viz_checkbox.stateChanged.connect(self.toggle_gpu_visualization)
        viz_options.addWidget(self.gpu_viz_checkbox)
        
        # Refresh button
        self.refresh_viz_btn = QPushButton("Refresh")
        self.refresh_viz_btn.clicked.connect(self.update_visualization)
        viz_options.addWidget(self.refresh_viz_btn)
        
        # Clear cache button
        self.clear_cache_btn = QPushButton("Clear Cache")
        self.clear_cache_btn.clicked.connect(self.clear_visualization_cache)
        viz_options.addWidget(self.clear_cache_btn)
        
        # Export visualization button
        self.export_viz_btn = QPushButton("Export Plot")
        self.export_viz_btn.clicked.connect(self.export_visualization)
        viz_options.addWidget(self.export_viz_btn)
        
        viz_layout.addLayout(viz_options)
        
        # Performance indicator
        self.viz_perf_label = QLabel("")
        viz_layout.addWidget(self.viz_perf_label)
        
        # Matplotlib figure with higher DPI for better quality
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        viz_layout.addWidget(self.canvas)
        
        self.viz_tab.setLayout(viz_layout)
        self.tabs.addTab(self.viz_tab, "Visualizations")
        
        # Apple ProRAW tab
        self.proraw_tab = QWidget()
        proraw_layout = QVBoxLayout()
        
        # Add Apple ProRAW panel to the dedicated tab
        self.apple_panel_main = AppleProRawPanel()
        proraw_layout.addWidget(self.apple_panel_main)
        
        # Add information about Apple ProRAW support
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        
        info_title = QLabel("About Apple ProRAW Support")
        info_title.setStyleSheet("font-size: 14px; font-weight: bold;")
        info_layout.addWidget(info_title)
        
        info_text = QLabel(
            "Apple ProRAW combines the benefits of computational photography with RAW. "
            "This tool extracts Apple-specific metadata including computational photography features "
            "such as Deep Fusion, Smart HDR, and Night Mode. The panel above shows which features "
            "were detected in the current image."
        )
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        
        compatibility_title = QLabel("Device Compatibility")
        compatibility_title.setStyleSheet("font-size: 14px; font-weight: bold;")
        info_layout.addWidget(compatibility_title)
        
        compatibility_text = QLabel(
            "ProRAW is available on iPhone 12 Pro, iPhone 12 Pro Max, and newer Pro models. "
            "Some computational photography features are specific to certain models."
        )
        compatibility_text.setWordWrap(True)
        info_layout.addWidget(compatibility_text)
        
        proraw_layout.addWidget(info_widget)
        proraw_layout.addStretch(1)
        
        self.proraw_tab.setLayout(proraw_layout)
        self.tabs.addTab(self.proraw_tab, "Apple ProRAW")
        
        # Hide the ProRAW tab by default - will show only when a ProRAW file is loaded
        self.tabs.setTabVisible(3, False)
        
        main_layout.addWidget(self.tabs)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
    
    def select_directory(self):
        """Select a directory of images to process"""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.status_label.setText(f"Selected directory: {directory}")
            self.process_directory(directory)
    
    def show_settings(self):
        """Show the settings dialog"""
        # Check if settings dialog is already open
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, ConfigDialog) and widget.isVisible():
                widget.activateWindow()  # Bring existing dialog to front
                return
        
        # Create and show new dialog if none exists
        self.config_dialog = ConfigDialog(self)
        
        # Connect the dialog's accepted signal to save preferences
        self.config_dialog.accepted.connect(self.save_preferences)
        
        self.config_dialog.show()
    
    def process_directory(self, directory_path):
        """Process a directory of images"""
        self.select_dir_btn.setEnabled(False)
        self.load_csv_btn.setEnabled(False)
        self.export_csv_btn.setEnabled(False)
        self.settings_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Processing...")
        
        # Start processing thread with all settings
        self.processing_thread = ProcessingThread(
            directory_path=directory_path, 
            max_image_size=self.max_image_size, 
            recursive=self.recursive, 
            skip_no_exif=self.skip_no_exif,
            cpu_cores=self.cpu_cores,
            memory_limit=self.memory_limit,
            use_gpu=self.use_gpu,
            gpu_image_processing=self.gpu_image_processing
        )
        
        # Connect signals
        self.processing_thread.progress_update.connect(self.update_progress)
        self.processing_thread.processing_complete.connect(self.processing_finished)
        
        # Show resource usage in status bar
        cores_info = f"Using {self.cpu_cores}/{self.total_cores} CPU cores"
        memory_info = f"Memory limit: {int(self.memory_limit * 100)}%" if self.memory_limit else "No memory limit"
        gpu_info = "GPU: Enabled" if self.use_gpu else "GPU: Disabled"
        
        self.status_label.setText(f"Processing... ({cores_info}, {memory_info}, {gpu_info})")
        
        # Start the thread
        self.processing_thread.start()
    
    def update_progress(self, progress, message):
        """Update progress bar and status message"""
        self.progress_bar.setValue(int(progress))
        self.status_label.setText(message)
    
    def processing_finished(self, exif_data):
        """Handle completion of processing"""
        self.select_dir_btn.setEnabled(True)
        self.load_csv_btn.setEnabled(True)
        self.settings_btn.setEnabled(True)
        
        if exif_data:
            self.export_csv_btn.setEnabled(True)
            self.field_selector_btn.setEnabled(True)
            
            # Show resource usage summary
            cores_info = f"Used {self.cpu_cores}/{self.total_cores} CPU cores"
            memory_info = f"Memory limit: {int(self.memory_limit * 100)}%" if self.memory_limit else "No memory limit"
            gpu_info = "GPU: Enabled" if self.use_gpu else "GPU: Disabled"
            
            status_text = f"Processed {len(exif_data)} images ({cores_info}, {memory_info}, {gpu_info})"
            
            # Check for Apple ProRAW files
            has_apple_proraw = False
            apple_count = 0
            for item in exif_data:
                if ('camera_make' in item and str(item['camera_make']).upper() == 'APPLE' and 
                    'file_type' in item and str(item['file_type']).upper() in ['DNG', 'PRORAW']):
                    has_apple_proraw = True
                    apple_count += 1
                    
                # Also check for Apple-specific fields
                apple_fields = [key for key in item.keys() if key.startswith('apple_')]
                if apple_fields and not has_apple_proraw:
                    has_apple_proraw = True
                    apple_count += 1
            
            # Show/hide Apple ProRAW tab based on presence of Apple files
            if hasattr(self, 'proraw_tab') and hasattr(self, 'tabs'):
                self.tabs.setTabVisible(3, has_apple_proraw)
                
                # If we have Apple ProRAW files, add to status text
                if has_apple_proraw:
                    status_text += f" | {apple_count} Apple ProRAW files detected"
            
            # Check if we have normalized fields available
            if self.processor.use_db and self.processor.db:
                try:
                    normalized_fields = self.processor.db.get_available_normalized_fields()
                    if normalized_fields:
                        status_text += f" | {len(normalized_fields)} normalized fields available"
                        print(f"Normalized EXIF fields available: {', '.join(normalized_fields)}")
                except Exception as e:
                    print(f"Error checking normalized fields: {e}")
            
            self.status_label.setText(status_text)
            
            self.processor.exif_data = exif_data
            self.analyzer.set_data(exif_data)
            self.update_data_table()
            self.update_summary()
            self.update_visualization()
        else:
            self.status_label.setText("No EXIF data found in the selected directory")
    
    def load_csv(self):
        """Load EXIF data from a CSV file"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load CSV File", "", "CSV Files (*.csv)")
        if file_path:
            if self.analyzer.load_csv(file_path):
                self.processor.load_from_csv(file_path)
                self.status_label.setText(f"Loaded data from {file_path}")
                self.export_csv_btn.setEnabled(True)
                self.update_data_table()
                self.update_summary()
                self.update_visualization()
            else:
                QMessageBox.warning(self, "Error", "Failed to load CSV file")
    
    def export_to_csv(self):
        """Export EXIF data to a CSV file"""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save CSV File", "", "CSV Files (*.csv)")
        if file_path:
            # Create export options dialog
            export_dialog = QDialog(self)
            export_dialog.setWindowTitle("Export Options")
            dialog_layout = QVBoxLayout()
            
            # Option for normalized fields
            normalized_checkbox = QCheckBox("Include normalized EXIF fields")
            normalized_checkbox.setChecked(True)
            normalized_checkbox.setToolTip("Normalized fields provide consistent access to metadata across different camera models")
            dialog_layout.addWidget(normalized_checkbox)
            
            # Option for Apple ProRAW fields
            apple_checkbox = QCheckBox("Include Apple ProRAW computational photography fields")
            apple_checkbox.setChecked(True)
            apple_checkbox.setToolTip("Include Apple-specific computational photography metadata")
            dialog_layout.addWidget(apple_checkbox)
            
            # Buttons
            button_layout = QHBoxLayout()
            ok_button = QPushButton("OK")
            ok_button.clicked.connect(export_dialog.accept)
            cancel_button = QPushButton("Cancel")
            cancel_button.clicked.connect(export_dialog.reject)
            button_layout.addWidget(ok_button)
            button_layout.addWidget(cancel_button)
            dialog_layout.addLayout(button_layout)
            
            export_dialog.setLayout(dialog_layout)
            
            # Show dialog
            if export_dialog.exec() != QDialog.DialogCode.Accepted:
                return
            
            include_normalized = normalized_checkbox.isChecked()
            include_apple = apple_checkbox.isChecked()
            
            if self.processor.save_to_csv(file_path, include_normalized=include_normalized):
                # If we're not including Apple fields, filter them out
                if not include_apple:
                    try:
                        # Read the CSV, filter out Apple fields, and write it back
                        df = pd.read_csv(file_path)
                        apple_columns = [col for col in df.columns if col.startswith('apple_')]
                        if apple_columns:
                            df = df.drop(columns=apple_columns)
                            df.to_csv(file_path, index=False)
                    except Exception as e:
                        print(f"Warning: Could not filter Apple fields: {e}")
                
                # Update status message
                status_parts = []
                if include_normalized:
                    status_parts.append("normalized fields")
                if include_apple:
                    status_parts.append("Apple ProRAW fields")
                
                status_msg = f"Exported data to {file_path}"
                if status_parts:
                    status_msg += f" (with {' and '.join(status_parts)})"
                
                self.status_label.setText(status_msg)
                QMessageBox.information(self, "Success", status_msg)
            else:
                QMessageBox.warning(self, "Error", "Failed to export CSV file")
    
    def update_data_table(self):
        """Update the data table with EXIF data"""
        # Temporarily disable sorting while updating the table
        self.data_table.setSortingEnabled(False)
        
        df = self.analyzer.df
        
        if df.empty:
            return
        
        # Check if we have normalized EXIF data to display
        has_normalized = False
        normalized_columns = []
        
        if 'normalized' in df.columns:
            has_normalized = True
            # Extract normalized fields from the first row that has them
            for i, row in df.iterrows():
                if pd.notna(row['normalized']) and isinstance(row['normalized'], dict):
                    normalized_columns = list(row['normalized'].keys())
                    break
        
        # Filter regular columns to only show selected fields
        all_columns = [col for col in df.columns if col != 'normalized' and col in self.selected_fields]
        
        # Add selected normalized columns with a prefix if available
        if has_normalized and normalized_columns:
            selected_norm_fields = [field for field in normalized_columns 
                                  if f"norm_{field}" in self.selected_fields]
            all_columns.extend([f"norm_{col}" for col in selected_norm_fields])
        
        # Set up table
        self.data_table.setRowCount(len(df))
        self.data_table.setColumnCount(len(all_columns))
        self.data_table.setHorizontalHeaderLabels(all_columns)
        
        # Check if any Apple ProRAW files are present
        has_apple_proraw = False
        apple_fields = []
        
        # Collect all Apple-specific fields
        for i, row in df.iterrows():
            # Check for Apple ProRAW files
            if ('camera_make' in row and str(row['camera_make']).upper() == 'APPLE' and 
                'file_type' in row and str(row['file_type']).upper() in ['DNG', 'PRORAW']):
                has_apple_proraw = True
            
            # Collect Apple-specific fields that are selected for display
            for col in [c for c in df.columns if c != 'normalized']:
                if col.startswith('apple_') and col not in apple_fields and col in self.selected_fields:
                    apple_fields.append(col)
                    has_apple_proraw = True
        
        # Show/hide Apple ProRAW tab based on presence of Apple files
        if hasattr(self, 'proraw_tab') and hasattr(self, 'tabs'):
            self.tabs.setTabVisible(3, has_apple_proraw)
            
            # Update the Apple ProRAW panel if we have one
            if has_apple_proraw and hasattr(self, 'apple_panel_main'):
                # Get the first Apple ProRAW file's data for the panel
                for i, row in df.iterrows():
                    if ('camera_make' in row and str(row['camera_make']).upper() == 'APPLE' and 
                        'file_type' in row and str(row['file_type']).upper() in ['DNG', 'PRORAW']):
                        # Convert row to dict for the panel
                        row_dict = row.to_dict()
                        self.apple_panel_main.update_panel(row_dict)
                        break
        
        # Fill table with data
        for i, row in df.iterrows():
            # Regular columns that are selected for display
            for j, col in enumerate(all_columns):
                # Handle normalized fields
                if col.startswith('norm_') and has_normalized:
                    norm_field = col[5:]  # Remove 'norm_' prefix
                    if pd.notna(row.get('normalized')) and isinstance(row['normalized'], dict) and norm_field in row['normalized']:
                        norm_data = row['normalized'][norm_field]
                        value = norm_data.get('value', '') if isinstance(norm_data, dict) else norm_data
                        item = QTableWidgetItem(str(value))
                        # Add tooltip showing the source field
                        if isinstance(norm_data, dict) and 'source_field' in norm_data:
                            item.setToolTip(f"Source field: {norm_data['source_field']}")
                        self.data_table.setItem(i, j, item)
                # Handle regular fields
                else:
                    if col in row:
                        item = QTableWidgetItem(str(row[col]))
                        
                        # Highlight Apple-specific fields
                        if col.startswith('apple_'):
                            item.setBackground(Qt.GlobalColor.lightGray)
                            item.setToolTip("Apple ProRAW specific field")
                        
                        self.data_table.setItem(i, j, item)
        
        self.data_table.resizeColumnsToContents()
        
        # Re-enable sorting after table is populated
        self.data_table.setSortingEnabled(True)
    
    def update_summary(self):
        """Update the summary tab with statistics"""
        stats = self.analyzer.get_summary_stats()
        
        if not stats:
            self.summary_label.setText("No data available for summary")
            return
        
        summary_text = f"""
        <h2>EXIF Data Summary</h2>
        <p><b>Total Images:</b> {stats['total_images']}</p>
        <p><b>Unique Camera Models:</b> {stats['unique_cameras']}</p>
        """
        
        if stats['date_range']:
            summary_text += f"<p><b>Date Range:</b> {stats['date_range'][0]} to {stats['date_range'][1]}</p>"
        
        self.summary_label.setText(summary_text)
    
    def update_visualization(self):
        """Update the visualization based on the selected type"""
        viz_type = self.viz_type_combo.currentText()
        
        # Check if data is available
        if self.analyzer.df is None or self.analyzer.df.empty:
            # If no data for this visualization, show a message
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"No data available for {viz_type}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14)
            self.figure.tight_layout()
            self.canvas.draw()
            self.viz_perf_label.setText("No data available")
            return
        
        # Start a timer to measure rendering performance
        start_time = time.time()
        
        # Use the enhanced visualizer with GPU acceleration if available
        try:
            # Create a copy of the dataframe to avoid unhashable type error
            df_copy = self.analyzer.df.copy()
            
            if viz_type == "Camera Distribution":
                fig = self.visualizer.plot_camera_distribution(df_copy, self.figure)
            elif viz_type == "Focal Length Distribution":
                fig = self.visualizer.plot_focal_length_distribution(df_copy, self.figure)
            elif viz_type == "Aperture Distribution":
                fig = self.visualizer.plot_aperture_distribution(df_copy, self.figure)
            elif viz_type == "ISO Distribution":
                fig = self.visualizer.plot_iso_distribution(df_copy, self.figure)
            elif viz_type == "Time of Day Distribution":
                fig = self.visualizer.plot_time_of_day(df_copy, self.figure)
            elif viz_type == "Location Map":
                fig = self.visualizer.plot_map(df_copy, self.figure)
            elif viz_type == "Aperture vs Focal Length":
                fig = self.visualizer.plot_heatmap(df_copy, "focal_length", "f_number", self.figure)
            elif viz_type == "ISO vs Time of Day":
                fig = self.visualizer.plot_heatmap(df_copy, "hour", "iso", self.figure)
            else:
                fig = None
            
            if fig:
                # Calculate and display rendering time
                render_time = time.time() - start_time
                perf_msg = f"Rendered {viz_type} in {render_time:.2f} seconds"
                if self.use_gpu:
                    perf_msg += " (GPU accelerated)"
                self.viz_perf_label.setText(perf_msg)
                print(perf_msg)
                
                self.canvas.draw()
            else:
                # If no data for this visualization, show a message
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, f"No data available for {viz_type}", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14)
                self.figure.tight_layout()
                self.canvas.draw()
                self.viz_perf_label.setText("No data available for this visualization")
        except Exception as e:
            # Handle any errors during visualization
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"Error creating visualization: {str(e)}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12, color='red')
            self.figure.tight_layout()
            self.canvas.draw()
            self.viz_perf_label.setText(f"Error: {str(e)}")
            print(f"Visualization error: {e}")
    
    def toggle_gpu_visualization(self):
        """Toggle GPU acceleration for visualizations"""
        self.use_gpu = self.gpu_viz_checkbox.isChecked()
        self.visualizer.use_gpu = self.use_gpu
        self.status_label.setText(f"GPU acceleration {'enabled' if self.use_gpu else 'disabled'} for visualizations")
        # Update the current visualization to reflect the change
        self.update_visualization()
    
    def clear_visualization_cache(self):
        """Clear the visualization cache"""
        self.visualizer.clear_cache()
        self.status_label.setText("Visualization cache cleared")
        self.viz_perf_label.setText("Cache cleared")
    
    def export_visualization(self):
        """Export the current visualization to a file"""
        viz_type = self.viz_type_combo.currentText()
        default_name = f"exif_{viz_type.lower().replace(' ', '_')}.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Visualization", default_name, "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)")
        
        if file_path:
            try:
                # Save the current figure to the specified file
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                self.status_label.setText(f"Visualization exported to {file_path}")
                QMessageBox.information(self, "Success", f"Visualization exported to {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to export visualization: {str(e)}")
                
    def show_field_selector(self):
        """Show dialog to select which fields to display"""
        # Create field selector dialog
        field_dialog = QDialog(self)
        field_dialog.setWindowTitle("Select Fields to Display")
        field_dialog.setMinimumWidth(400)
        field_dialog.setMinimumHeight(500)
        
        layout = QVBoxLayout()
        
        # Instructions
        instructions = QLabel("Select fields to display in the data table:")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Get all available fields from the data
        all_fields = []
        if not self.analyzer.df.empty:
            # Regular fields
            all_fields = [col for col in self.analyzer.df.columns if col != 'normalized']
            
            # Add normalized fields if available
            if 'normalized' in self.analyzer.df.columns:
                for i, row in self.analyzer.df.iterrows():
                    if pd.notna(row['normalized']) and isinstance(row['normalized'], dict):
                        norm_fields = list(row['normalized'].keys())
                        all_fields.extend([f"norm_{field}" for field in norm_fields])
                        break
        
        # Remove duplicates and sort
        all_fields = sorted(list(set(all_fields)))
        
        # Create scrollable area for checkboxes
        scroll_area = QWidget()
        scroll_layout = QVBoxLayout(scroll_area)
        
        # Add checkboxes for each field
        self.field_checkboxes = {}
        
        # Add "Select All" checkbox
        select_all_cb = QCheckBox("Select/Deselect All")
        layout.addWidget(select_all_cb)
        
        # Group fields by categories
        field_categories = {
            "Basic": ['file_name', 'file_path', 'file_type', 'file_size'],
            "Camera": ['camera_make', 'camera_model', 'lens_model'],
            "Exposure": ['focal_length', 'f_number', 'iso', 'exposure_time', 'shutter_speed', 
                        'exposure_program', 'metering_mode', 'white_balance', 'flash'],
            "Image": ['width', 'height', 'orientation', 'scene_type', 'scene_capture_type'],
            "Location": ['gps_latitude', 'gps_longitude', 'altitude', 'location'],
            "Time": ['date_time', 'create_date', 'modify_date'],
            "Normalized": [f for f in all_fields if f.startswith('norm_')],
            "Apple": [f for f in all_fields if f.startswith('apple_')],
            "Other": []
        }
        
        # Categorize remaining fields
        for field in all_fields:
            categorized = False
            for category, fields in field_categories.items():
                if field in fields:
                    categorized = True
                    break
            if not categorized and not field.startswith('norm_') and not field.startswith('apple_'):
                field_categories["Other"].append(field)
        
        # Create section for each category
        for category, fields in field_categories.items():
            if not fields:  # Skip empty categories
                continue
                
            # Add category header
            category_label = QLabel(f"<b>{category}</b>")
            scroll_layout.addWidget(category_label)
            
            # Add checkboxes for fields in this category
            for field in sorted(fields):
                if field in all_fields:  # Only add if field exists in data
                    cb = QCheckBox(field)
                    cb.setChecked(field in self.selected_fields)
                    self.field_checkboxes[field] = cb
                    scroll_layout.addWidget(cb)
            
            # Add a small spacer after each category
            spacer = QWidget()
            spacer.setFixedHeight(10)
            scroll_layout.addWidget(spacer)
        
        # Create scrollable area
        scroll_widget = QScrollArea()
        scroll_widget.setWidgetResizable(True)
        scroll_widget.setWidget(scroll_area)
        layout.addWidget(scroll_widget)
        
        # Connect select all checkbox
        def toggle_all(state):
            for cb in self.field_checkboxes.values():
                cb.setChecked(state == Qt.CheckState.Checked)
        
        select_all_cb.stateChanged.connect(toggle_all)
        
        # Add buttons
        button_layout = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        cancel_btn = QPushButton("Cancel")
        
        button_layout.addWidget(apply_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        # Set dialog layout
        field_dialog.setLayout(layout)
        
        # Connect buttons
        def apply_selection():
            self.selected_fields = [field for field, cb in self.field_checkboxes.items() 
                                  if cb.isChecked()]
            self.update_data_table()
            # Save selected fields to preferences
            self.save_preferences()
            field_dialog.accept()
        
        apply_btn.clicked.connect(apply_selection)
        cancel_btn.clicked.connect(field_dialog.reject)
        
        # Show dialog
        field_dialog.exec()



def main():
    """Main entry point for the application"""
    app = QApplication(sys.argv)
    window = ExifToolGUI()
    window.show()
    sys.exit(app.exec())
    
    
# Add methods for loading and saving preferences
def load_preferences(self):
    """Load user preferences from file"""
    try:
        if os.path.exists(self.preferences_file):
            with open(self.preferences_file, 'r') as f:
                preferences = json.load(f)
                
                # Load selected fields if available
                if 'selected_fields' in preferences:
                    self.selected_fields = preferences['selected_fields']
                else:
                    # Use default visualizable fields if not found
                    self.selected_fields = self.visualizable_fields.copy()
                    
                # Load other preferences if needed
                if 'max_image_size' in preferences:
                    self.max_image_size = preferences['max_image_size']
                if 'recursive' in preferences:
                    self.recursive = preferences['recursive']
                if 'skip_no_exif' in preferences:
                    self.skip_no_exif = preferences['skip_no_exif']
                if 'use_gpu' in preferences:
                    self.use_gpu = preferences['use_gpu'] and GPU_AVAILABLE
                if 'gpu_image_processing' in preferences:
                    self.gpu_image_processing = preferences['gpu_image_processing'] and GPU_AVAILABLE
                if 'gpu_data_analysis' in preferences:
                    self.gpu_data_analysis = preferences['gpu_data_analysis'] and GPU_AVAILABLE
                if 'cpu_cores' in preferences:
                    self.cpu_cores = min(preferences['cpu_cores'], self.total_cores)
                if 'memory_limit' in preferences:
                    self.memory_limit = preferences['memory_limit']
                
                print(f"Loaded preferences from {self.preferences_file}")
        else:
            # Use default visualizable fields if no preferences file
            self.selected_fields = self.visualizable_fields.copy()
    except Exception as e:
        print(f"Error loading preferences: {e}")
        # Use default visualizable fields if error
        self.selected_fields = self.visualizable_fields.copy()

def save_preferences(self):
    """Save user preferences to file"""
    try:
        # Create preferences directory if it doesn't exist
        os.makedirs(os.path.dirname(self.preferences_file), exist_ok=True)
        
        # Prepare preferences dictionary
        preferences = {
            'selected_fields': self.selected_fields,
            'max_image_size': self.max_image_size,
            'recursive': self.recursive,
            'skip_no_exif': self.skip_no_exif,
            'use_gpu': self.use_gpu,
            'gpu_image_processing': self.gpu_image_processing,
            'gpu_data_analysis': self.gpu_data_analysis,
            'cpu_cores': self.cpu_cores,
            'memory_limit': self.memory_limit
        }
        
        # Save to file
        with open(self.preferences_file, 'w') as f:
            json.dump(preferences, f, indent=2)
            
        print(f"Saved preferences to {self.preferences_file}")
    except Exception as e:
        print(f"Error saving preferences: {e}")

# Add methods to ExifToolGUI class
ExifToolGUI.load_preferences = load_preferences
ExifToolGUI.save_preferences = save_preferences

# Add method for handling column header clicks and sorting
def on_header_clicked(self, column_index):
    """Handle column header click to sort the table"""
    # Get the column name
    column_name = self.data_table.horizontalHeaderItem(column_index).text()
    
    # Toggle sort order for this column
    if column_index in self.sort_orders:
        self.sort_orders[column_index] = not self.sort_orders[column_index]
    else:
        self.sort_orders[column_index] = True  # Default to ascending first
    
    # Determine sort order
    sort_order = Qt.SortOrder.AscendingOrder if self.sort_orders[column_index] else Qt.SortOrder.DescendingOrder
    
    # Sort the table
    self.data_table.sortItems(column_index, sort_order)
    
    # Update status message
    direction = "ascending" if self.sort_orders[column_index] else "descending"
    self.status_label.setText(f"Sorted by {column_name} ({direction})")

ExifToolGUI.on_header_clicked = on_header_clicked


if __name__ == "__main__":
    main()
