# EXIF Tool for macOS

A Python-based tool designed to process and visualize EXIF data from photos on macOS (Apple Silicon).

## Features

- Ingest directories of photos
- Index photos based on their EXIF data using SQLite database
- Create and manage collections of photos
- Export EXIF data to CSV files for further processing
- Provide graphical visualization of EXIF data
- Optimized for Apple Silicon with Metal GPU acceleration
- Configurable resource management (CPU cores, memory limits, GPU acceleration)

## Requirements

- macOS running on Apple Silicon
- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main application:

```
python exif_tool.py
```

Or use the command line interface for batch processing:

```
python exif_cli.py --input /path/to/photos --output /path/to/output --visualize
```

Database operations:

```
# Create a collection
python exif_cli.py --input /path/to/photos --collection "My Collection"

# List all collections
python exif_cli.py --list-collections

# Show database statistics
python exif_cli.py --stats

# Export a collection to CSV
python exif_cli.py --export-collection 1 --output /path/to/output.csv

# Search the database
python exif_cli.py --query "camera_make=SONY"

## License

MIT
