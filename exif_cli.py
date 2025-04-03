#!/usr/bin/env python3
"""
EXIF Tool CLI - Command-line interface for processing and exporting EXIF data
"""

import os
import sys
import argparse
import warnings
import datetime
from PIL import Image, ImageFile
# Disable DecompressionBombWarning
Image.MAX_IMAGE_PIXELS = None  # Disable the DecompressionBombWarning
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Handle truncated images
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
from exif_tool import ExifProcessor, ExifAnalyzer
from exif_db import ExifDatabase
import matplotlib.pyplot as plt
from tqdm import tqdm


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='EXIF Tool - Process and analyze EXIF data from photos')
    
    # Input/Output options
    parser.add_argument('--input', '-i', help='Input directory containing photos')
    parser.add_argument('--output', '-o', help='Output directory for CSV and visualizations')
    parser.add_argument('--recursive', '-r', action='store_true', help='Process subdirectories recursively')
    
    # Database options
    parser.add_argument('--use-db', action='store_true', default=True, help='Use SQLite database for storing EXIF data (default: True)')
    parser.add_argument('--no-db', action='store_false', dest='use_db', help='Disable database storage')
    parser.add_argument('--db-path', help='Custom path for SQLite database file')
    parser.add_argument('--collection', help='Custom name for the collection in the database')
    parser.add_argument('--query', help='Search query for the database (e.g., "camera_make=SONY")')
    parser.add_argument('--export-collection', type=int, help='Export a specific collection ID to CSV')
    parser.add_argument('--list-collections', action='store_true', help='List all collections in the database')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    
    # Visualization options
    parser.add_argument('--visualize', '-v', action='store_true', help='Generate visualization charts')
    parser.add_argument('--format', choices=['png', 'pdf', 'jpg'], default='png', help='Output format for visualizations')
    
    # Resource management
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration if available')
    parser.add_argument('--cores', type=int, help='Number of CPU cores to use (default: n-2 on Apple Silicon)')
    parser.add_argument('--memory', type=int, default=75, help='Memory limit as percentage of total RAM (default: 75)')
    
    return parser.parse_args()


def main():
    """Main entry point for the CLI application"""
    args = parse_args()
    
    # Initialize database if enabled
    db = None
    if args.use_db:
        try:
            db = ExifDatabase(args.db_path)
            print(f"Using EXIF database at: {db.db_path}")
            
            # Handle database-specific operations
            if args.list_collections:
                list_collections(db)
                return 0
                
            if args.stats:
                show_stats(db)
                return 0
                
            if args.export_collection is not None:
                if not args.output:
                    print("Error: --output is required when using --export-collection")
                    return 1
                export_collection(db, args.export_collection, args.output)
                return 0
                
            if args.query:
                if not args.output:
                    print("Error: --output is required when using --query")
                    return 1
                search_database(db, args.query, args.output)
                return 0
        except Exception as e:
            print(f"Database error: {e}")
            if args.use_db:
                print("Continuing without database support")
    
    # Check if input directory is required for the requested operation
    if not args.input:
        print("Error: --input is required for processing images")
        print("Use --list-collections, --stats, --export-collection, or --query for database operations without input")
        return 1
    
    # Validate input directory for processing
    if not os.path.isdir(args.input):
        print(f"Error: Input directory '{args.input}' does not exist")
        return 1
    
    # Create output directory if specified and doesn't exist
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    print(f"\nProcessing photos in: {args.input}")
    print(f"Recursive mode: {args.recursive}")
    
    # Count files before processing
    total_files = 0
    image_files = []
    for root, _, files in os.walk(args.input):
        if not args.recursive and root != args.input:
            continue
        for file in files:
            total_files += 1
            # Check if it's an image file with supported extension
            ext = os.path.splitext(file)[1].lower()
            if ext in ['.jpg', '.jpeg', '.tiff', '.tif', '.png', '.heic', '.heif', '.nef', '.cr2', '.arw']:
                image_files.append(os.path.join(root, file))
    
    print(f"Supported extensions: ['.jpg', '.jpeg', '.tiff', '.tif', '.png', '.heic', '.heif', '.nef', '.cr2', '.arw']")
    print(f"Checking directory: {args.input} ({total_files} files)")
    for image in image_files:
        print(f"Found image: {image}")
    print(f"Total files checked: {total_files}")
    print(f"Found {len(image_files)} image files with supported extensions")
    
    if not image_files:
        print("No supported image files found in the specified directory")
        return 1
    
    # Process photos
    processor = ExifProcessor(
        use_db=args.use_db,
        db_path=args.db_path
    )
    
    # Process the directory
    exif_data = processor.process_directory(args.input)
    
    if not exif_data:
        print("No EXIF data extracted from the specified directory")
        return 1
    
    print(f"\nProcessed {len(exif_data)} photos with EXIF data")
    
    # Save to CSV if output directory is specified
    if args.output:
        csv_path = os.path.join(args.output, 'exif_data.csv')
        if processor.save_to_csv(csv_path):
            print(f"EXIF data saved to {csv_path}")
        else:
            print("Error saving CSV file")
    
    # Generate visualizations if requested and output directory is specified
    if args.visualize and args.output:
        print("\nGenerating visualizations...")
        analyzer = ExifAnalyzer(exif_data)
        
        # Create visualizations directory
        viz_dir = os.path.join(args.output, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Generate and save visualizations
        visualizations = [
            ('camera_distribution', analyzer.plot_camera_distribution),
            ('focal_length_distribution', analyzer.plot_focal_length_distribution),
            ('aperture_distribution', analyzer.plot_aperture_distribution),
            ('iso_distribution', analyzer.plot_iso_distribution),
            ('time_of_day_distribution', analyzer.plot_time_of_day)
        ]
        
        for name, plot_func in visualizations:
            try:
                fig = plot_func()
                if fig:
                    output_path = os.path.join(viz_dir, f"{name}.{args.format}")
                    fig.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print(f"Saved {output_path}")
            except Exception as e:
                print(f"Error generating {name}: {e}")
        
        print(f"Visualizations saved to {viz_dir}")
    
    # Close database connection if used
    if db:
        db.close()
    
    print("\nProcessing complete!")
    return 0


def list_collections(db):
    """List all collections in the database"""
    try:
        db.cursor.execute("SELECT id, name, description, date_created FROM collections ORDER BY date_created DESC")
        collections = db.cursor.fetchall()
        
        if not collections:
            print("No collections found in the database")
            return
        
        print("\nCollections in database:")
        print("-" * 80)
        print(f"{'ID':<5} {'Name':<30} {'Created':<20} {'Images':<10} Description")
        print("-" * 80)
        
        for collection in collections:
            # Count images in this collection
            db.cursor.execute("SELECT COUNT(*) FROM collection_images WHERE collection_id = ?", (collection['id'],))
            image_count = db.cursor.fetchone()[0]
            
            print(f"{collection['id']:<5} {collection['name'][:30]:<30} {collection['date_created']:<20} {image_count:<10} {collection['description']}")
    except Exception as e:
        print(f"Error listing collections: {e}")


def show_stats(db):
    """Show database statistics"""
    try:
        stats = db.get_stats()
        
        print("\nDatabase Statistics:")
        print(f"Total images: {stats['total_images']}")
        
        if stats['images_by_camera_make']:
            print("\nImages by camera make:")
            for make, count in stats['images_by_camera_make'].items():
                print(f"  {make}: {count}")
        
        if stats['images_by_file_type']:
            print("\nImages by file type:")
            for file_type, count in stats['images_by_file_type'].items():
                print(f"  {file_type}: {count}")
        
        if stats['images_by_year']:
            print("\nImages by year:")
            for year, count in stats['images_by_year'].items():
                print(f"  {year}: {count}")
    except Exception as e:
        print(f"Error getting database stats: {e}")


def export_collection(db, collection_id, output_dir):
    """Export a collection to CSV"""
    try:
        # Verify collection exists
        db.cursor.execute("SELECT name FROM collections WHERE id = ?", (collection_id,))
        result = db.cursor.fetchone()
        if not result:
            print(f"Error: Collection with ID {collection_id} not found")
            return
        
        collection_name = result[0]
        print(f"Exporting collection '{collection_name}' (ID: {collection_id})")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Export to CSV
        csv_path = os.path.join(output_dir, f"collection_{collection_id}_{collection_name}.csv")
        db.export_to_csv(csv_path, collection_id)
        
        print(f"Exported to {csv_path}")
    except Exception as e:
        print(f"Error exporting collection: {e}")


def search_database(db, query, output_dir):
    """Search the database and export results"""
    try:
        # Parse query string (format: field=value)
        if '=' not in query:
            print("Error: Query must be in format 'field=value' (e.g., 'camera_make=SONY')")
            return
        
        field, value = query.split('=', 1)
        field = field.strip()
        value = value.strip()
        
        print(f"Searching for {field}={value}")
        
        # Build search parameters based on field
        search_params = {}
        if field == 'camera_make':
            search_params['camera_make'] = value
        elif field == 'camera_model':
            search_params['camera_model'] = value
        elif field == 'lens_model':
            search_params['lens_model'] = value
        elif field == 'file_type':
            search_params['file_type'] = value
        elif field == 'date_from':
            search_params['date_from'] = value
        elif field == 'date_to':
            search_params['date_to'] = value
        elif field == 'tag_name':
            search_params['tag_name'] = value
        else:
            print(f"Error: Unsupported search field '{field}'")
            print("Supported fields: camera_make, camera_model, lens_model, file_type, date_from, date_to, tag_name")
            return
        
        # Perform search
        results = db.search_images(**search_params)
        
        if not results:
            print("No matching images found")
            return
        
        print(f"Found {len(results)} matching images")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Export results to CSV
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(output_dir, f"search_{field}_{value}_{timestamp}.csv")
        
        # Convert results to DataFrame and save to CSV
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        
        print(f"Exported results to {csv_path}")
    except Exception as e:
        print(f"Error searching database: {e}")


if __name__ == "__main__":
    sys.exit(main())
