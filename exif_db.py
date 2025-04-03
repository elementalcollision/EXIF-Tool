#!/usr/bin/env python3
"""
EXIF Database Module
Provides a lightweight SQLite database for storing EXIF data
"""

import os
import sqlite3
import json
import datetime
import pandas as pd
from typing import Dict, List, Any, Optional, Union


class ExifDatabase:
    """SQLite database for storing EXIF data from images"""
    
    def __init__(self, db_path: str = None):
        """Initialize the database
        
        Args:
            db_path: Path to the SQLite database file. If None, uses default location.
        """
        if db_path is None:
            # Use default location in user's home directory
            home_dir = os.path.expanduser("~")
            db_dir = os.path.join(home_dir, ".exif_tool")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "exif_data.db")
        
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
        # Connect to database
        self._connect()
        
        # Create tables if they don't exist
        self._create_tables()
        
        print(f"EXIF database initialized at: {self.db_path}")
    
    def _connect(self):
        """Connect to the SQLite database"""
        self.conn = sqlite3.connect(self.db_path)
        # Enable foreign keys
        self.conn.execute("PRAGMA foreign_keys = ON")
        # Use Row factory for better column access
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        # Images table - stores basic image information
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE NOT NULL,
            file_name TEXT NOT NULL,
            file_size INTEGER,
            file_type TEXT,
            width INTEGER,
            height INTEGER,
            date_taken TEXT,
            date_added TEXT,
            camera_make TEXT,
            camera_model TEXT,
            lens_model TEXT,
            focal_length REAL,
            f_number REAL,
            exposure_time TEXT,
            iso INTEGER,
            has_gps BOOLEAN,
            latitude REAL,
            longitude REAL,
            altitude REAL,
            last_updated TEXT
        )
        ''')
        
        # EXIF data table - stores all EXIF data as key-value pairs
        # This allows for flexible storage of any EXIF tag
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS exif_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            tag_name TEXT NOT NULL,
            tag_value TEXT,
            tag_type TEXT,
            FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
            UNIQUE(image_id, tag_name)
        )
        ''')
        
        # Normalized EXIF data table - stores standardized EXIF fields across different camera models
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS normalized_exif_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            normalized_field TEXT NOT NULL,
            field_value TEXT,
            value_type TEXT,
            source_field TEXT,  -- Original vendor-specific field name
            FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
            UNIQUE(image_id, normalized_field)
        )
        ''')
        
        # Collections table - for organizing images into collections
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS collections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            date_created TEXT,
            date_modified TEXT
        )
        ''')
        
        # Collection_images table - many-to-many relationship
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS collection_images (
            collection_id INTEGER,
            image_id INTEGER,
            date_added TEXT,
            PRIMARY KEY (collection_id, image_id),
            FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE,
            FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
        )
        ''')
        
        # Create indexes for performance
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_file_path ON images(file_path)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_date_taken ON images(date_taken)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_camera ON images(camera_make, camera_model)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_exif_tag ON exif_data(tag_name)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_exif_image_id ON exif_data(image_id)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_normalized_exif_field ON normalized_exif_data(normalized_field)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_normalized_exif_image_id ON normalized_exif_data(image_id)')
        
        # Commit changes
        self.conn.commit()
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    def _normalize_exif_fields(self, exif_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Normalize vendor-specific EXIF fields to standard field names
        
        Args:
            exif_data: Dictionary containing EXIF data
            
        Returns:
            Dictionary of normalized fields with their values and source fields
        """
        normalized_fields = {}
        
        # Define field normalization mappings
        # Format: {normalized_field: [list of possible source fields]}
        field_mappings = {
            'Firmware': ['firmware', 'camera_firmware', 'sony_firmware', 'canon_firmware', 'nikon_firmware', 
                        'fuji_firmware', 'leica_firmware', 'olympus_firmware', 'panasonic_firmware',
                        'Image FirmwareVersion', 'MakerNote FirmwareVersion', 'software'],
            'SerialNumber': ['camera_serial', 'sony_camera_serial', 'canon_serial_number', 'nikon_serial_number',
                           'fuji_serial_number', 'leica_serial_number', 'body_serial_number', 'serial_number'],
            'LensModel': ['lens_model', 'sony_lens_model', 'canon_lens_model', 'nikon_lens_model',
                        'fuji_lens_model', 'leica_lens_model', 'lens_name', 'lens_info'],
            'LensSerialNumber': ['lens_serial', 'sony_lens_serial', 'canon_lens_serial', 'nikon_lens_serial',
                               'fuji_lens_serial', 'leica_lens_serial'],
            'ShutterCount': ['shutter_count', 'sony_shutter_count', 'canon_shutter_count', 'nikon_shutter_count',
                           'fuji_shutter_count', 'leica_shutter_count', 'exposure_count', 'image_count'],
            'ColorProfile': ['color_profile', 'sony_color_profile', 'canon_color_profile', 'nikon_color_profile',
                           'fuji_color_profile', 'leica_color_profile', 'color_space', 'color_mode'],
            'WhiteBalance': ['white_balance', 'sony_white_balance', 'canon_white_balance', 'nikon_white_balance',
                           'fuji_white_balance', 'leica_white_balance', 'wb_mode'],
            'ExposureMode': ['exposure_mode', 'sony_exposure_mode', 'canon_exposure_mode', 'nikon_exposure_mode',
                           'fuji_exposure_mode', 'leica_exposure_mode', 'exposure_program'],
            'FocusMode': ['focus_mode', 'sony_focus_mode', 'canon_focus_mode', 'nikon_focus_mode',
                        'fuji_focus_mode', 'leica_focus_mode', 'af_mode', 'focus_type'],
            'ImageStabilization': ['image_stabilization', 'sony_steadyshot', 'canon_is', 'nikon_vr',
                                 'fuji_ois', 'leica_ois', 'stabilization', 'is_mode'],
            'DriveMode': ['drive_mode', 'sony_drive_mode', 'canon_drive_mode', 'nikon_drive_mode',
                        'fuji_drive_mode', 'leica_drive_mode', 'continuous_mode', 'shooting_mode'],
            'CreativeStyle': ['creative_style', 'sony_creative_style', 'canon_picture_style', 'nikon_picture_control',
                            'fuji_film_simulation', 'leica_film_mode', 'picture_style', 'film_mode'],
            'NoiseReduction': ['noise_reduction', 'sony_noise_reduction', 'canon_noise_reduction', 'nikon_noise_reduction',
                             'fuji_noise_reduction', 'leica_noise_reduction', 'nr_setting', 'high_iso_nr']
        }
        
        # Process each normalized field
        for normalized_field, source_fields in field_mappings.items():
            # Find the first matching source field
            for source_field in source_fields:
                if source_field in exif_data and exif_data[source_field] is not None:
                    # Determine value type
                    value = exif_data[source_field]
                    if isinstance(value, (int, float, bool)):
                        value_type = type(value).__name__
                    elif isinstance(value, (list, dict, tuple)):
                        value_type = 'json'
                    else:
                        value_type = 'string'
                    
                    # Store the normalized field with its value and source
                    normalized_fields[normalized_field] = {
                        'value': str(value),
                        'type': value_type,
                        'source': source_field
                    }
                    break
        
        return normalized_fields
    
    def add_image(self, exif_data: Dict[str, Any]) -> int:
        """Add or update an image and its EXIF data in the database
        
        Args:
            exif_data: Dictionary containing EXIF data
            
        Returns:
            image_id: ID of the added/updated image
        """
        if not exif_data or 'file_path' not in exif_data:
            raise ValueError("EXIF data must contain 'file_path'")
        
        # Extract core image fields
        file_path = exif_data.get('file_path')
        file_name = exif_data.get('file_name', os.path.basename(file_path))
        file_size = exif_data.get('file_size', 0)
        file_type = exif_data.get('file_type', '')
        width = exif_data.get('width', 0)
        height = exif_data.get('height', 0)
        date_taken = exif_data.get('date_taken', '')
        camera_make = exif_data.get('camera_make', '')
        camera_model = exif_data.get('camera_model', '')
        lens_model = exif_data.get('lens_model', '')
        focal_length = self._extract_numeric(exif_data.get('focal_length', '0'))
        f_number = self._extract_numeric(exif_data.get('f_number', '0'))
        exposure_time = exif_data.get('exposure_time', '')
        iso = self._extract_numeric(exif_data.get('iso', '0'))
        
        # GPS data
        has_gps = 'gps_latitude' in exif_data and 'gps_longitude' in exif_data
        latitude = self._extract_numeric(exif_data.get('gps_latitude', '0'))
        longitude = self._extract_numeric(exif_data.get('gps_longitude', '0'))
        altitude = self._extract_numeric(exif_data.get('gps_altitude', '0'))
        
        # Current timestamp
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Check if image already exists
        self.cursor.execute("SELECT id FROM images WHERE file_path = ?", (file_path,))
        result = self.cursor.fetchone()
        
        if result:
            # Update existing image
            image_id = result[0]
            self.cursor.execute('''
            UPDATE images SET
                file_name = ?, file_size = ?, file_type = ?,
                width = ?, height = ?,
                date_taken = ?, camera_make = ?, camera_model = ?,
                lens_model = ?, focal_length = ?, f_number = ?,
                exposure_time = ?, iso = ?,
                has_gps = ?, latitude = ?, longitude = ?, altitude = ?,
                last_updated = ?
            WHERE id = ?
            ''', (
                file_name, file_size, file_type,
                width, height,
                date_taken, camera_make, camera_model,
                lens_model, focal_length, f_number,
                exposure_time, iso,
                has_gps, latitude, longitude, altitude,
                now, image_id
            ))
            
            # Delete existing EXIF data for this image
            self.cursor.execute("DELETE FROM exif_data WHERE image_id = ?", (image_id,))
            self.cursor.execute("DELETE FROM normalized_exif_data WHERE image_id = ?", (image_id,))
        else:
            # Insert new image
            self.cursor.execute('''
            INSERT INTO images (
                file_path, file_name, file_size, file_type,
                width, height,
                date_taken, date_added, camera_make, camera_model,
                lens_model, focal_length, f_number,
                exposure_time, iso,
                has_gps, latitude, longitude, altitude,
                last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                file_path, file_name, file_size, file_type,
                width, height,
                date_taken, now, camera_make, camera_model,
                lens_model, focal_length, f_number,
                exposure_time, iso,
                has_gps, latitude, longitude, altitude,
                now
            ))
            
            # Get the ID of the newly inserted image
            image_id = self.cursor.lastrowid
        
        # Insert all EXIF data as key-value pairs
        for key, value in exif_data.items():
            # Skip None values
            if value is None:
                continue
                
            # Determine value type
            if isinstance(value, (int, float, bool)):
                tag_type = type(value).__name__
                tag_value = str(value)
            elif isinstance(value, (list, dict, tuple)):
                tag_type = 'json'
                tag_value = json.dumps(value)
            else:
                tag_type = 'string'
                tag_value = str(value)
            
            self.cursor.execute('''
            INSERT INTO exif_data (image_id, tag_name, tag_value, tag_type)
            VALUES (?, ?, ?, ?)
            ''', (image_id, key, tag_value, tag_type))
        
        # Process and insert normalized EXIF fields
        normalized_fields = self._normalize_exif_fields(exif_data)
        for field_name, field_data in normalized_fields.items():
            self.cursor.execute('''
            INSERT INTO normalized_exif_data (image_id, normalized_field, field_value, value_type, source_field)
            VALUES (?, ?, ?, ?, ?)
            ''', (image_id, field_name, field_data['value'], field_data['type'], field_data['source']))
        
        # Commit changes
        self.conn.commit()
        
        return image_id
    
    def get_image(self, image_id: int = None, file_path: str = None, include_normalized: bool = True) -> Dict[str, Any]:
        """Get image data by ID or file path
        
        Args:
            image_id: ID of the image
            file_path: Path to the image file
            include_normalized: Whether to include normalized EXIF fields
            
        Returns:
            Dictionary containing image data and EXIF tags
        """
        if image_id is None and file_path is None:
            raise ValueError("Either image_id or file_path must be provided")
        
        if file_path is not None:
            # Get image by file path
            self.cursor.execute("SELECT id FROM images WHERE file_path = ?", (file_path,))
            result = self.cursor.fetchone()
            if not result:
                return None
            image_id = result[0]
        
        # Get image data
        self.cursor.execute("SELECT * FROM images WHERE id = ?", (image_id,))
        image_row = self.cursor.fetchone()
        if not image_row:
            return None
        
        # Convert row to dictionary
        image_data = {key: image_row[key] for key in image_row.keys()}
        
        # Get EXIF data
        self.cursor.execute("SELECT tag_name, tag_value, tag_type FROM exif_data WHERE image_id = ?", (image_id,))
        exif_rows = self.cursor.fetchall()
        
        # Add EXIF data to image data
        for row in exif_rows:
            tag_name, tag_value, tag_type = row
            
            # Convert value based on type
            if tag_type == 'int':
                value = int(tag_value)
            elif tag_type == 'float':
                value = float(tag_value)
            elif tag_type == 'bool':
                value = tag_value.lower() in ('true', '1', 'yes')
            elif tag_type == 'json':
                value = json.loads(tag_value)
            else:
                value = tag_value
            
            image_data[tag_name] = value
        
        # Get normalized EXIF data if requested
        if include_normalized:
            self.cursor.execute("""
            SELECT normalized_field, field_value, value_type, source_field 
            FROM normalized_exif_data 
            WHERE image_id = ?
            """, (image_id,))
            
            normalized_rows = self.cursor.fetchall()
            
            # Add normalized data to image data
            if normalized_rows:
                normalized_data = {}
                for row in normalized_rows:
                    normalized_field, field_value, value_type, source_field = row
                    
                    # Convert value based on type
                    if value_type == 'int':
                        value = int(field_value) if field_value.isdigit() else field_value
                    elif value_type == 'float':
                        try:
                            value = float(field_value)
                        except ValueError:
                            value = field_value
                    elif value_type == 'bool':
                        value = field_value.lower() in ('true', '1', 'yes')
                    elif value_type == 'json':
                        try:
                            value = json.loads(field_value)
                        except json.JSONDecodeError:
                            value = field_value
                    else:
                        value = field_value
                    
                    # Add to normalized data dictionary
                    normalized_data[normalized_field] = {
                        'value': value,
                        'source_field': source_field
                    }
                
                # Add normalized data to image data
                image_data['normalized'] = normalized_data
        
        return image_data
    
    def delete_image(self, image_id: int = None, file_path: str = None) -> bool:
        """Delete an image and its EXIF data from the database
        
        Args:
            image_id: ID of the image
            file_path: Path to the image file
            
        Returns:
            True if successful, False otherwise
        """
        if image_id is None and file_path is None:
            raise ValueError("Either image_id or file_path must be provided")
        
        if file_path is not None:
            # Get image by file path
            self.cursor.execute("SELECT id FROM images WHERE file_path = ?", (file_path,))
            result = self.cursor.fetchone()
            if not result:
                return False
            image_id = result[0]
        
        # Delete image (cascade will delete related EXIF data)
        self.cursor.execute("DELETE FROM images WHERE id = ?", (image_id,))
        self.conn.commit()
        
        return self.cursor.rowcount > 0
    
    def create_collection(self, name: str, description: str = "") -> int:
        """Create a new collection
        
        Args:
            name: Name of the collection
            description: Description of the collection
            
        Returns:
            ID of the created collection
        """
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Check if collection already exists
        self.cursor.execute("SELECT id FROM collections WHERE name = ?", (name,))
        result = self.cursor.fetchone()
        
        if result:
            return result[0]
        
        # Insert new collection
        self.cursor.execute('''
        INSERT INTO collections (name, description, date_created, date_modified)
        VALUES (?, ?, ?, ?)
        ''', (name, description, now, now))
        
        self.conn.commit()
        return self.cursor.lastrowid
    
    def add_image_to_collection(self, collection_id: int, image_id: int) -> bool:
        """Add an image to a collection
        
        Args:
            collection_id: ID of the collection
            image_id: ID of the image
            
        Returns:
            True if successful, False otherwise
        """
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            self.cursor.execute('''
            INSERT OR IGNORE INTO collection_images (collection_id, image_id, date_added)
            VALUES (?, ?, ?)
            ''', (collection_id, image_id, now))
            
            self.conn.commit()
            return True
        except sqlite3.Error:
            return False
    
    def get_collection_images(self, collection_id: int) -> List[Dict[str, Any]]:
        """Get all images in a collection
        
        Args:
            collection_id: ID of the collection
            
        Returns:
            List of image data dictionaries
        """
        self.cursor.execute('''
        SELECT i.* FROM images i
        JOIN collection_images ci ON i.id = ci.image_id
        WHERE ci.collection_id = ?
        ORDER BY i.date_taken
        ''', (collection_id,))
        
        rows = self.cursor.fetchall()
        return [{key: row[key] for key in row.keys()} for row in rows]
    
    def search_images(self, 
                      camera_make: str = None,
                      camera_model: str = None,
                      date_from: str = None,
                      date_to: str = None,
                      lens_model: str = None,
                      min_focal_length: float = None,
                      max_focal_length: float = None,
                      min_aperture: float = None,
                      max_aperture: float = None,
                      min_iso: int = None,
                      max_iso: int = None,
                      file_type: str = None,
                      has_gps: bool = None,
                      tag_name: str = None,
                      tag_value: str = None,
                      normalized_field: str = None,
                      normalized_value: str = None) -> List[Dict[str, Any]]:
        """Search for images based on criteria
        
        Args:
            camera_make: Filter by camera manufacturer
            camera_model: Filter by camera model
            date_from: Filter by date taken (from)
            date_to: Filter by date taken (to)
            lens_model: Filter by lens model
            min_focal_length: Filter by minimum focal length
            max_focal_length: Filter by maximum focal length
            min_aperture: Filter by minimum aperture
            max_aperture: Filter by maximum aperture
            min_iso: Filter by minimum ISO
            max_iso: Filter by maximum ISO
            file_type: Filter by file type
            has_gps: Filter by presence of GPS data
            tag_name: Filter by specific EXIF tag name
            tag_value: Filter by specific EXIF tag value
            normalized_field: Filter by normalized EXIF field name
            normalized_value: Filter by normalized EXIF field value
            
        Returns:
            List of matching image data dictionaries
        """
        query = "SELECT DISTINCT i.* FROM images i"
        params = []
        where_clauses = []
        
        # Join with exif_data if searching by tag
        if tag_name is not None:
            query += " JOIN exif_data e ON i.id = e.image_id"
            where_clauses.append("e.tag_name = ?")
            params.append(tag_name)
            
            if tag_value is not None:
                where_clauses.append("e.tag_value LIKE ?")
                params.append(f"%{tag_value}%")
                
        # Join with normalized_exif_data if searching by normalized field
        if normalized_field is not None:
            query += " JOIN normalized_exif_data n ON i.id = n.image_id"
            where_clauses.append("n.normalized_field = ?")
            params.append(normalized_field)
            
            if normalized_value is not None:
                where_clauses.append("n.field_value LIKE ?")
                params.append(f"%{normalized_value}%")
        
        # Add where clauses for each search parameter
        if camera_make is not None:
            where_clauses.append("i.camera_make LIKE ?")
            params.append(f"%{camera_make}%")
        
        if camera_model is not None:
            where_clauses.append("i.camera_model LIKE ?")
            params.append(f"%{camera_model}%")
        
        if date_from is not None:
            where_clauses.append("i.date_taken >= ?")
            params.append(date_from)
        
        if date_to is not None:
            where_clauses.append("i.date_taken <= ?")
            params.append(date_to)
        
        if lens_model is not None:
            where_clauses.append("i.lens_model LIKE ?")
            params.append(f"%{lens_model}%")
        
        if min_focal_length is not None:
            where_clauses.append("i.focal_length >= ?")
            params.append(min_focal_length)
        
        if max_focal_length is not None:
            where_clauses.append("i.focal_length <= ?")
            params.append(max_focal_length)
        
        if min_aperture is not None:
            where_clauses.append("i.f_number >= ?")
            params.append(min_aperture)
        
        if max_aperture is not None:
            where_clauses.append("i.f_number <= ?")
            params.append(max_aperture)
        
        if min_iso is not None:
            where_clauses.append("i.iso >= ?")
            params.append(min_iso)
        
        if max_iso is not None:
            where_clauses.append("i.iso <= ?")
            params.append(max_iso)
        
        if file_type is not None:
            where_clauses.append("i.file_type LIKE ?")
            params.append(f"%{file_type}%")
        
        if has_gps is not None:
            where_clauses.append("i.has_gps = ?")
            params.append(1 if has_gps else 0)
        
        # Combine where clauses
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        
        # Order by date taken
        query += " ORDER BY i.date_taken DESC"
        
        # Execute query
        self.cursor.execute(query, params)
        rows = self.cursor.fetchall()
        
        return [{key: row[key] for key in row.keys()} for row in rows]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics
        
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        # Total images
        self.cursor.execute("SELECT COUNT(*) FROM images")
        stats['total_images'] = self.cursor.fetchone()[0]
        
        # Images by camera make
        self.cursor.execute('''
        SELECT camera_make, COUNT(*) as count 
        FROM images 
        WHERE camera_make != '' 
        GROUP BY camera_make 
        ORDER BY count DESC
        ''')
        stats['images_by_camera_make'] = {row[0]: row[1] for row in self.cursor.fetchall()}
        
        # Images by file type
        self.cursor.execute('''
        SELECT file_type, COUNT(*) as count 
        FROM images 
        GROUP BY file_type 
        ORDER BY count DESC
        ''')
        stats['images_by_file_type'] = {row[0]: row[1] for row in self.cursor.fetchall()}
        
        # Images by year
        self.cursor.execute('''
        SELECT substr(date_taken, 1, 4) as year, COUNT(*) as count 
        FROM images 
        WHERE date_taken != '' 
        GROUP BY year 
        ORDER BY year
        ''')
        stats['images_by_year'] = {row[0]: row[1] for row in self.cursor.fetchall()}
        
        return stats
    
    def export_to_csv(self, output_path: str, collection_id: int = None, include_normalized: bool = True) -> str:
        """Export EXIF data to CSV
        
        Args:
            output_path: Path to save the CSV file
            collection_id: Optional collection ID to filter images
            include_normalized: Whether to include normalized EXIF fields
            
        Returns:
            Path to the saved CSV file
        """
        # Build query
        query = '''
        SELECT i.*, GROUP_CONCAT(e.tag_name || ':' || e.tag_value, '|') as exif_tags
        FROM images i
        LEFT JOIN exif_data e ON i.id = e.image_id
        '''
        
        params = []
        if collection_id is not None:
            query += '''
            JOIN collection_images ci ON i.id = ci.image_id
            WHERE ci.collection_id = ?
            '''
            params.append(collection_id)
        
        query += 'GROUP BY i.id'
        
        # Execute query
        self.cursor.execute(query, params)
        rows = self.cursor.fetchall()
        
        # Convert to DataFrame
        data = [{key: row[key] for key in row.keys()} for row in rows]
        df = pd.DataFrame(data)
        
        # Add normalized EXIF data if requested
        if include_normalized:
            # Get all available normalized fields
            self.cursor.execute("""
            SELECT DISTINCT normalized_field 
            FROM normalized_exif_data 
            ORDER BY normalized_field
            """)
            normalized_fields = [row[0] for row in self.cursor.fetchall()]
            
            # For each image, get its normalized fields
            for i, row in enumerate(data):
                image_id = row['id']
                
                self.cursor.execute("""
                SELECT normalized_field, field_value 
                FROM normalized_exif_data 
                WHERE image_id = ?
                """, (image_id,))
                
                normalized_data = {field: None for field in normalized_fields}
                for field, value in self.cursor.fetchall():
                    normalized_data[field] = value
                
                # Add normalized fields as columns with 'norm_' prefix
                for field, value in normalized_data.items():
                    df.at[i, f'norm_{field}'] = value
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def get_normalized_field_values(self, field_name: str) -> List[str]:
        """Get all unique values for a normalized field
        
        Args:
            field_name: Name of the normalized field
            
        Returns:
            List of unique values for the field
        """
        self.cursor.execute("""
        SELECT DISTINCT field_value 
        FROM normalized_exif_data 
        WHERE normalized_field = ? 
        ORDER BY field_value
        """, (field_name,))
        
        return [row[0] for row in self.cursor.fetchall()]
    
    def get_available_normalized_fields(self) -> List[str]:
        """Get all available normalized fields in the database
        
        Returns:
            List of normalized field names
        """
        self.cursor.execute("""
        SELECT DISTINCT normalized_field 
        FROM normalized_exif_data 
        ORDER BY normalized_field
        """)
        
        return [row[0] for row in self.cursor.fetchall()]
    
    def get_images_by_normalized_field(self, field_name: str, field_value: str = None) -> List[Dict[str, Any]]:
        """Get images that have a specific normalized field
        
        Args:
            field_name: Name of the normalized field
            field_value: Optional value to filter by
            
        Returns:
            List of image data dictionaries
        """
        query = """
        SELECT i.* FROM images i
        JOIN normalized_exif_data n ON i.id = n.image_id
        WHERE n.normalized_field = ?
        """
        
        params = [field_name]
        
        if field_value is not None:
            query += " AND n.field_value LIKE ?"
            params.append(f"%{field_value}%")
            
        query += " ORDER BY i.date_taken DESC"
        
        self.cursor.execute(query, params)
        rows = self.cursor.fetchall()
        
        return [{key: row[key] for key in row.keys()} for row in rows]
    
    def _extract_numeric(self, value: Union[str, int, float]) -> float:
        """Extract numeric value from string
        
        Args:
            value: String or numeric value
            
        Returns:
            Numeric value
        """
        if isinstance(value, (int, float)):
            return float(value)
        
        if not value or not isinstance(value, str):
            return 0.0
        
        # Handle fractions like "1/100"
        if '/' in value:
            try:
                num, denom = value.split('/')
                return float(num) / float(denom)
            except (ValueError, ZeroDivisionError):
                pass
        
        # Try direct conversion
        try:
            return float(value)
        except ValueError:
            return 0.0


# Test function
def test_database():
    """Test the EXIF database"""
    # Create database in memory for testing
    db = ExifDatabase(":memory:")
    
    # Sample EXIF data
    exif_data = {
        'file_path': '/path/to/image.jpg',
        'file_name': 'image.jpg',
        'file_size': 1024000,
        'file_type': 'JPEG',
        'width': 3000,
        'height': 2000,
        'date_taken': '2025-01-01 12:00:00',
        'camera_make': 'SONY',
        'camera_model': 'ILCE-7RM5',
        'lens_model': 'FE 24-70mm F2.8 GM',
        'focal_length': '50',
        'f_number': '2.8',
        'exposure_time': '1/100',
        'iso': '100',
        'gps_latitude': '37.7749',
        'gps_longitude': '-122.4194',
        'gps_altitude': '10',
        'raw_type': 'ARW',
        'sony_raw_black_level': 200,
        'sony_raw_white_level': 16383,
        # Add vendor-specific firmware version field
        'sony_firmware_version': '1.2.3'
    }
    
    # Add image to database
    image_id = db.add_image(exif_data)
    print(f"Added image with ID: {image_id}")
    
    # Retrieve image with normalized fields
    retrieved_data = db.get_image(image_id, include_normalized=True)
    print(f"Retrieved {len(retrieved_data)} fields")
    
    # Check if normalized data is present
    if 'normalized' in retrieved_data:
        print(f"Normalized fields: {list(retrieved_data['normalized'].keys())}")
    
    # Create collection
    collection_id = db.create_collection("Test Collection", "Collection for testing")
    print(f"Created collection with ID: {collection_id}")
    
    # Add image to collection
    db.add_image_to_collection(collection_id, image_id)
    
    # Search for images by camera make
    results = db.search_images(camera_make="SONY")
    print(f"Found {len(results)} images by SONY")
    
    # Search for images by normalized field
    if db.get_available_normalized_fields():
        normalized_field = db.get_available_normalized_fields()[0]
        normalized_results = db.search_images(normalized_field=normalized_field)
        print(f"Found {len(normalized_results)} images with normalized field '{normalized_field}'")
    
    # Get all available normalized fields
    normalized_fields = db.get_available_normalized_fields()
    print(f"Available normalized fields: {normalized_fields}")
    
    # Get values for a normalized field if available
    if normalized_fields:
        field_values = db.get_normalized_field_values(normalized_fields[0])
        print(f"Values for '{normalized_fields[0]}': {field_values}")
    
    # Get stats
    stats = db.get_stats()
    print(f"Database stats: {stats}")
    
    # Close database
    db.close()
    print("Database test completed successfully")


if __name__ == "__main__":
    test_database()
