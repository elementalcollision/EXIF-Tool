#!/usr/bin/env python3
"""
Test script for verifying all camera extractors in the GUI
This script automates testing the EXIF tool GUI with different camera RAW formats
"""

import os
import sys
import time
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtTest import QTest
from exif_tool import ExifToolGUI

class ExifToolTester:
    """Class for testing the EXIF tool GUI with different camera RAW formats"""
    
    def __init__(self, test_dir):
        """Initialize the tester with the test directory"""
        self.test_dir = test_dir
        self.app = QApplication(sys.argv)
        self.window = None
        
        # Get list of test files
        self.test_files = []
        if os.path.exists(test_dir):
            for file in os.listdir(test_dir):
                if file.lower().endswith(('.arw', '.nef', '.cr2', '.cr3', '.dng', '.raf', '.rw2')):
                    self.test_files.append(os.path.join(test_dir, file))
        
        # Sort test files by extension
        self.test_files.sort(key=lambda x: os.path.splitext(x)[1].lower())
        
        print(f"Found {len(self.test_files)} test files in {test_dir}")
        for file in self.test_files:
            print(f"  - {os.path.basename(file)}")
    
    def start_gui(self):
        """Start the EXIF tool GUI"""
        self.window = ExifToolGUI()
        self.window.show()
        
        # Set resource settings to use n-2 cores and 75% RAM
        self.window.cpu_cores = max(1, os.cpu_count() - 2)
        self.window.memory_limit_percent = 75
        
        # Enable GPU acceleration if available
        self.window.use_gpu = True
        
        print(f"GUI started with {self.window.cpu_cores} CPU cores, "
              f"{self.window.memory_limit_percent}% memory limit, "
              f"GPU: {self.window.use_gpu}")
        
        # Schedule the test to run after GUI is fully loaded
        QTimer.singleShot(1000, self.run_tests)
        
        # Start the event loop
        return self.app.exec()
    
    def run_tests(self):
        """Run tests for each camera RAW format"""
        if not self.test_files:
            print("No test files found. Exiting.")
            self.window.close()
            return
        
        print("\nStarting tests for all camera extractors...")
        
        # Process each test file
        for i, file_path in enumerate(self.test_files):
            file_name = os.path.basename(file_path)
            print(f"\nTesting file {i+1}/{len(self.test_files)}: {file_name}")
            
            # Process the file
            self.process_file(file_path)
            
            # Wait for processing to complete
            QTest.qWait(2000)
            
            # Check results
            self.check_results(file_path)
            
            # Wait before next file
            QTest.qWait(1000)
        
        print("\nAll tests completed!")
        
        # Close the window after tests
        QTimer.singleShot(2000, self.window.close)
    
    def process_file(self, file_path):
        """Process a single file in the GUI"""
        print(f"Processing: {os.path.basename(file_path)}")
        
        # Directly load the file using the processor
        self.window.processor.extract_exif(file_path)
        
        # Update the UI with the results
        self.window.update_image_preview(file_path)
        self.window.update_exif_table()
        
        print(f"File processed: {os.path.basename(file_path)}")
    
    def check_results(self, file_path):
        """Check the results for a processed file"""
        file_name = os.path.basename(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        
        # Get the EXIF data from the processor
        exif_data = self.window.processor.current_exif_data
        
        if not exif_data:
            print(f"ERROR: No EXIF data found for {file_name}")
            return
        
        # Check camera make and model
        camera_make = exif_data.get('camera_make', 'Unknown')
        camera_model = exif_data.get('camera_model', 'Unknown')
        print(f"Camera: {camera_make} {camera_model}")
        
        # Check for camera-specific fields
        prefixes = {
            'SONY': 'sony_',
            'NIKON': 'nikon_',
            'CANON': 'canon_',
            'LEICA': 'leica_',
            'FUJI': 'fuji_',
            'FUJIFILM': 'fuji_',
            'APPLE': 'apple_',
            'PANASONIC': 'panasonic_',
            'LUMIX': 'panasonic_'
        }
        
        # Get the prefix for this camera
        prefix = None
        for make, p in prefixes.items():
            if make in camera_make.upper():
                prefix = p
                break
        
        if not prefix and '.DNG' in ext.upper():
            prefix = 'dng_'
        
        # Count camera-specific fields
        if prefix:
            specific_fields = [key for key in exif_data.keys() if key.startswith(prefix)]
            print(f"Found {len(specific_fields)} {prefix.strip('_')}-specific fields")
            
            # Print some examples
            if specific_fields:
                print(f"Example fields:")
                for field in sorted(specific_fields)[:5]:  # Show first 5 fields
                    value = exif_data[field]
                    if isinstance(value, (list, dict)) and len(str(value)) > 100:
                        value = str(value)[:100] + "..."
                    print(f"  {field}: {value}")
                
                if len(specific_fields) > 5:
                    print(f"  ... and {len(specific_fields) - 5} more fields")
                
                print(f"TEST PASSED: {file_name} - {camera_make} {camera_model}")
            else:
                print(f"TEST FAILED: No {prefix.strip('_')}-specific fields found for {file_name}")
        else:
            print(f"TEST FAILED: No camera-specific fields found for {file_name}")

def main():
    """Main function"""
    # Set test directory
    test_dir = "Test_for_EXIF"
    
    # Create and run the tester
    tester = ExifToolTester(test_dir)
    return tester.start_gui()

if __name__ == "__main__":
    sys.exit(main())
