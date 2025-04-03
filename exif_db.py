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
        
        # Commit changes
        self.conn.commit()
        
        return image_id
    
    def get_image(self, image_id: int = None, file_path: str = None) -> Dict[str, Any]:
        """Get image data by ID or file path
        
        Args:
            image_id: ID of the image
            file_path: Path to the image file
            
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
                      tag_value: str = None) -> List[Dict[str, Any]]:
        """Search for images based on criteria
        
        Args:
            Various search criteria
            
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
    
    def export_to_csv(self, output_path: str, collection_id: int = None) -> str:
        """Export EXIF data to CSV
        
        Args:
            output_path: Path to save the CSV file
            collection_id: Optional collection ID to filter images
            
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
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        return output_path
    
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
        'sony_raw_white_level': 16383
    }
    
    # Add image to database
    image_id = db.add_image(exif_data)
    print(f"Added image with ID: {image_id}")
    
    # Retrieve image
    retrieved_data = db.get_image(image_id)
    print(f"Retrieved {len(retrieved_data)} fields")
    
    # Create collection
    collection_id = db.create_collection("Test Collection", "Collection for testing")
    print(f"Created collection with ID: {collection_id}")
    
    # Add image to collection
    db.add_image_to_collection(collection_id, image_id)
    
    # Search for images
    results = db.search_images(camera_make="SONY")
    print(f"Found {len(results)} images by SONY")
    
    # Get stats
    stats = db.get_stats()
    print(f"Database stats: {stats}")
    
    # Close database
    db.close()
    print("Database test completed successfully")


if __name__ == "__main__":
    test_database()
