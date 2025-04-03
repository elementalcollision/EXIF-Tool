#!/usr/bin/env python3
"""
Resource Settings Dialog for EXIF Tool
Provides a dialog to configure resource management settings
"""

import os
import platform
import multiprocessing
import psutil
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, 
                            QFormLayout, QLabel, QSpinBox, QSlider, QCheckBox,
                            QPushButton, QLineEdit, QFileDialog)
from PyQt6.QtCore import Qt

class ResourceSettingsDialog(QDialog):
    """Resource management settings dialog for EXIF Tool"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Resource Settings")
        self.setMinimumWidth(500)
        
        # Get system info
        self.total_cores = multiprocessing.cpu_count()
        self.total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        self.is_apple_silicon = platform.processor() == 'arm'
        
        # Create layout
        layout = QVBoxLayout()
        
        # System info display
        info_group = QGroupBox("System Information")
        info_layout = QFormLayout()
        
        # Show processor info
        processor_label = QLabel(f"Processor: {platform.processor()}")
        if self.is_apple_silicon:
            processor_label.setText(processor_label.text() + " (Apple Silicon)")
        info_layout.addRow(processor_label)
        
        # Show core count
        cores_label = QLabel(f"Total CPU Cores: {self.total_cores}")
        info_layout.addRow(cores_label)
        
        # Show memory info
        memory_label = QLabel(f"Total Memory: {self.total_memory_gb:.1f} GB")
        info_layout.addRow(memory_label)
        
        # Show GPU info
        gpu_available = False
        gpu_info = "Not available"
        
        try:
            import torch
            if torch.backends.mps.is_available():
                gpu_available = True
                gpu_info = "Metal Performance Shaders (MPS) available"
        except ImportError:
            pass
        
        gpu_label = QLabel(f"GPU Acceleration: {gpu_info}")
        info_layout.addRow(gpu_label)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # CPU settings
        cpu_group = QGroupBox("CPU Settings")
        cpu_layout = QFormLayout()
        
        # Default to n-2 cores on Apple Silicon, n-1 on others
        default_cores = max(1, self.total_cores - 2) if self.is_apple_silicon else max(1, self.total_cores - 1)
        current_cores = parent.cpu_cores if parent and hasattr(parent, 'cpu_cores') else default_cores
        
        self.cpu_cores_spin = QSpinBox()
        self.cpu_cores_spin.setMinimum(1)
        self.cpu_cores_spin.setMaximum(self.total_cores)
        self.cpu_cores_spin.setValue(current_cores)
        cpu_layout.addRow("CPU Cores to Use:", self.cpu_cores_spin)
        
        # Add recommendation label
        if self.is_apple_silicon:
            cpu_recommendation = QLabel(f"Recommended: {default_cores} cores (n-2 for Apple Silicon)")
        else:
            cpu_recommendation = QLabel(f"Recommended: {default_cores} cores (n-1 for this processor)")
        cpu_recommendation.setStyleSheet("color: gray; font-style: italic;")
        cpu_layout.addRow("", cpu_recommendation)
        
        cpu_group.setLayout(cpu_layout)
        layout.addWidget(cpu_group)
        
        # Memory settings
        memory_group = QGroupBox("Memory Settings")
        memory_layout = QFormLayout()
        
        # Default to 75% memory limit
        current_memory_limit = parent.memory_limit_percent if parent and hasattr(parent, 'memory_limit_percent') else 75
        
        self.memory_limit_slider = QSlider(Qt.Orientation.Horizontal)
        self.memory_limit_slider.setMinimum(10)
        self.memory_limit_slider.setMaximum(90)
        self.memory_limit_slider.setValue(current_memory_limit)
        self.memory_limit_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.memory_limit_slider.setTickInterval(10)
        
        self.memory_limit_label = QLabel(f"{current_memory_limit}% ({current_memory_limit/100 * self.total_memory_gb:.1f} GB)")
        self.memory_limit_slider.valueChanged.connect(self.update_memory_label)
        
        memory_layout.addRow("Memory Usage Limit:", self.memory_limit_slider)
        memory_layout.addRow("", self.memory_limit_label)
        
        # Add recommendation label
        memory_recommendation = QLabel("Recommended: 75% of available RAM")
        memory_recommendation.setStyleSheet("color: gray; font-style: italic;")
        memory_layout.addRow("", memory_recommendation)
        
        memory_group.setLayout(memory_layout)
        layout.addWidget(memory_group)
        
        # GPU settings
        gpu_group = QGroupBox("GPU Acceleration")
        gpu_layout = QFormLayout()
        
        current_gpu_enabled = parent.use_gpu if parent and hasattr(parent, 'use_gpu') else gpu_available
        
        self.gpu_checkbox = QCheckBox("Enable GPU Acceleration")
        self.gpu_checkbox.setChecked(current_gpu_enabled)
        self.gpu_checkbox.setEnabled(gpu_available)
        gpu_layout.addRow(self.gpu_checkbox)
        
        if not gpu_available:
            gpu_note = QLabel("Metal GPU acceleration not available on this system")
            gpu_note.setStyleSheet("color: red;")
            gpu_layout.addRow("", gpu_note)
        else:
            gpu_note = QLabel("Uses Metal Performance Shaders on Apple Silicon")
            gpu_note.setStyleSheet("color: gray; font-style: italic;")
            gpu_layout.addRow("", gpu_note)
        
        gpu_group.setLayout(gpu_layout)
        layout.addWidget(gpu_group)
        
        # Database settings
        db_group = QGroupBox("Database Settings")
        db_layout = QFormLayout()
        
        current_db_enabled = parent.use_db if parent and hasattr(parent, 'use_db') else True
        current_db_path = parent.db_path if parent and hasattr(parent, 'db_path') else None
        
        self.db_checkbox = QCheckBox("Enable SQLite Database")
        self.db_checkbox.setChecked(current_db_enabled)
        db_layout.addRow(self.db_checkbox)
        
        self.db_path_edit = QLineEdit()
        if current_db_path:
            self.db_path_edit.setText(current_db_path)
        else:
            # Default location
            home_dir = os.path.expanduser("~")
            default_db_path = os.path.join(home_dir, ".exif_tool", "exif_data.db")
            self.db_path_edit.setText(default_db_path)
        
        self.db_path_edit.setEnabled(current_db_enabled)
        self.db_checkbox.toggled.connect(lambda checked: self.db_path_edit.setEnabled(checked))
        
        db_path_layout = QHBoxLayout()
        db_path_layout.addWidget(self.db_path_edit)
        
        db_path_btn = QPushButton("Browse...")
        db_path_btn.clicked.connect(self.browse_db_path)
        db_path_layout.addWidget(db_path_btn)
        
        db_layout.addRow("Database Path:", db_path_layout)
        
        db_group.setLayout(db_layout)
        layout.addWidget(db_group)
        
        # Buttons
        button_box = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        button_box.addWidget(save_btn)
        button_box.addWidget(cancel_btn)
        layout.addLayout(button_box)
        
        self.setLayout(layout)
    
    def update_memory_label(self, value):
        """Update the memory limit label when the slider changes"""
        self.memory_limit_label.setText(f"{value}% ({value/100 * self.total_memory_gb:.1f} GB)")
    
    def browse_db_path(self):
        """Browse for database path"""
        current_path = self.db_path_edit.text()
        directory = os.path.dirname(current_path) if current_path else os.path.expanduser("~")
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Database File",
            directory,
            "SQLite Database (*.db);;All Files (*)"
        )
        
        if file_path:
            self.db_path_edit.setText(file_path)
    
    def get_settings(self):
        """Get the resource settings"""
        return {
            "cpu_cores": self.cpu_cores_spin.value(),
            "memory_limit_percent": self.memory_limit_slider.value(),
            "use_gpu": self.gpu_checkbox.isChecked(),
            "use_db": self.db_checkbox.isChecked(),
            "db_path": self.db_path_edit.text() if self.db_checkbox.isChecked() else None
        }
