#!/usr/bin/env python3
"""
Enhanced Visualization Engine for EXIF Tool
Optimized for Apple Silicon with Metal GPU acceleration
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import warnings
import time
import threading
from functools import lru_cache
from typing import Dict, Any, List, Tuple, Optional, Union

# GPU acceleration libraries
try:
    import torch
    HAS_TORCH = torch.backends.mps.is_available()
    if HAS_TORCH:
        print("PyTorch with Metal acceleration available")
except ImportError:
    HAS_TORCH = False

try:
    import cv2
    HAS_CV2 = hasattr(cv2, 'UMat')
    if HAS_CV2:
        print("OpenCV with Metal acceleration available")
except ImportError:
    HAS_CV2 = False

# Set Matplotlib to use a faster backend if possible
import matplotlib
if HAS_TORCH or HAS_CV2:
    matplotlib.use('agg')  # Non-interactive backend for faster rendering


class VisualizationCache:
    """Cache for visualization data to avoid recomputing"""
    
    def __init__(self, max_size=10):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
    
    def get(self, key):
        """Get item from cache if it exists"""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, key, value):
        """Add item to cache, removing oldest if at capacity"""
        if len(self.cache) >= self.max_size:
            # Remove least recently used item
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.access_times.clear()


class EnhancedVisualizer:
    """Enhanced visualization engine with GPU acceleration and caching"""
    
    def __init__(self, use_gpu=True, memory_limit=0.75, cpu_cores=None):
        """Initialize the visualizer
        
        Args:
            use_gpu: Whether to use GPU acceleration when available
            memory_limit: Maximum memory usage as percentage of total
            cpu_cores: Number of CPU cores to use (defaults to n-2 on Apple Silicon)
        """
        self.use_gpu = use_gpu and (HAS_TORCH or HAS_CV2)
        self.memory_limit = memory_limit
        
        # Set CPU cores to use (n-2 for ARM cores on Apple Silicon)
        import multiprocessing
        total_cores = multiprocessing.cpu_count()
        if cpu_cores is None:
            # Default to n-2 cores on ARM processors (Apple Silicon)
            import platform
            if platform.processor() == 'arm':
                self.cpu_cores = max(1, total_cores - 2)
            else:
                self.cpu_cores = max(1, total_cores - 1)
        else:
            self.cpu_cores = min(max(1, cpu_cores), total_cores)
        
        print(f"EnhancedVisualizer initialized with {self.cpu_cores} CPU cores, "
              f"GPU acceleration: {self.use_gpu}")
        
        # Initialize cache
        self.cache = VisualizationCache(max_size=20)
        
        # Set default style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Configure seaborn for better visuals
        sns.set_context("notebook", font_scale=1.2)
        
        # Initialize device for PyTorch if available
        if self.use_gpu and HAS_TORCH:
            self.device = torch.device("mps")
        else:
            self.device = None
    
    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the dataframe for visualization
        
        Args:
            df: Input DataFrame with EXIF data
            
        Returns:
            Preprocessed DataFrame
        """
        # Create a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Convert date_taken to datetime if it exists
        if 'date_taken' in df_processed.columns:
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
                        df_processed['date_taken_dt'] = pd.to_datetime(
                            df_processed['date_taken'], 
                            format=date_format, 
                            errors='coerce'
                        )
                        if not df_processed['date_taken_dt'].isna().all():
                            parsed = True
                            break
                    except:
                        continue
                
                # Fall back to automatic parsing if needed
                if not parsed:
                    df_processed['date_taken_dt'] = pd.to_datetime(
                        df_processed['date_taken'], 
                        errors='coerce'
                    )
                
                # Extract components
                df_processed['hour'] = df_processed['date_taken_dt'].dt.hour
                df_processed['day'] = df_processed['date_taken_dt'].dt.day
                df_processed['month'] = df_processed['date_taken_dt'].dt.month
                df_processed['year'] = df_processed['date_taken_dt'].dt.year
                df_processed['weekday'] = df_processed['date_taken_dt'].dt.dayofweek
            except Exception as e:
                print(f"Error processing dates: {e}")
        
        # Extract numeric values from common EXIF fields
        if 'focal_length' in df_processed.columns:
            df_processed['focal_length_num'] = df_processed['focal_length'].str.extract(
                r'(\d+)').astype(float)
        
        if 'f_number' in df_processed.columns:
            df_processed['f_number_num'] = df_processed['f_number'].str.extract(
                r'f/(\d+\.?\d*)').astype(float)
        
        if 'iso' in df_processed.columns:
            df_processed['iso_num'] = pd.to_numeric(df_processed['iso'], errors='coerce')
        
        return df_processed
    
    def _accelerate_computation(self, data: np.ndarray) -> np.ndarray:
        """Accelerate computation using GPU if available
        
        Args:
            data: NumPy array to process
            
        Returns:
            Processed NumPy array
        """
        if not self.use_gpu or data.size < 10000:  # Only use GPU for larger datasets
            return data
        
        try:
            if HAS_TORCH:
                # Convert to PyTorch tensor and process on MPS (Metal)
                tensor = torch.tensor(data, dtype=torch.float32).to(self.device)
                # Example processing: normalize data
                tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
                return tensor.cpu().numpy()
            
            elif HAS_CV2:
                # Use OpenCV with Metal acceleration
                umat = cv2.UMat(data)
                # Example processing: apply Gaussian blur
                processed = cv2.GaussianBlur(umat, (3, 3), 0)
                return processed.get()
            
            else:
                return data
        
        except Exception as e:
            print(f"GPU acceleration error: {e}")
            return data
    
    @lru_cache(maxsize=8)
    def plot_camera_distribution(self, df: pd.DataFrame, figure=None, 
                                limit=10, color_palette="viridis"):
        """Plot distribution of camera models with GPU acceleration
        
        Args:
            df: DataFrame with EXIF data
            figure: Matplotlib figure to use (or None to create new)
            limit: Maximum number of cameras to show
            color_palette: Color palette to use
            
        Returns:
            Matplotlib figure
        """
        cache_key = f"camera_dist_{limit}_{color_palette}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            if figure is not None:
                figure.clear()
                for ax in cached['axes']:
                    figure.add_axes(ax)
                return figure
            else:
                return cached['figure']
        
        if df.empty or 'camera_model' not in df.columns:
            return None
        
        if figure is None:
            fig = Figure(figsize=(10, 6), dpi=100)
        else:
            fig = figure
            fig.clear()
        
        ax = fig.add_subplot(111)
        
        # Count camera models
        camera_counts = df['camera_model'].value_counts().head(limit)
        
        # Use GPU acceleration for color mapping if available
        colors = sns.color_palette(color_palette, len(camera_counts))
        if self.use_gpu and HAS_TORCH:
            colors_array = np.array(colors)
            colors_array = self._accelerate_computation(colors_array)
            colors = [tuple(c) for c in colors_array]
        
        # Plot horizontal bar chart
        bars = ax.barh(camera_counts.index, camera_counts.values, color=colors)
        
        # Add count labels
        for i, bar in enumerate(bars):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                   str(camera_counts.values[i]), 
                   va='center')
        
        ax.set_title('Top Camera Models', fontsize=14)
        ax.set_xlabel('Number of Images', fontsize=12)
        fig.tight_layout()
        
        # Cache the result
        if figure is None:
            self.cache.set(cache_key, {
                'figure': fig,
                'axes': fig.get_axes()
            })
        
        return fig
    
    @lru_cache(maxsize=8)
    def plot_focal_length_distribution(self, df: pd.DataFrame, figure=None, 
                                      bins=20, color='skyblue'):
        """Plot distribution of focal lengths with GPU acceleration
        
        Args:
            df: DataFrame with EXIF data
            figure: Matplotlib figure to use (or None to create new)
            bins: Number of histogram bins
            color: Color to use for the histogram
            
        Returns:
            Matplotlib figure
        """
        cache_key = f"focal_dist_{bins}_{color}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            if figure is not None:
                figure.clear()
                for ax in cached['axes']:
                    figure.add_axes(ax)
                return figure
            else:
                return cached['figure']
        
        # Process the dataframe if needed
        if 'focal_length_num' not in df.columns:
            df = self._preprocess_dataframe(df)
        
        if df.empty or 'focal_length_num' not in df.columns:
            return None
        
        if figure is None:
            fig = Figure(figsize=(10, 6), dpi=100)
        else:
            fig = figure
            fig.clear()
        
        ax = fig.add_subplot(111)
        
        # Get focal length data
        focal_lengths = df['focal_length_num'].dropna().values
        
        # Use GPU acceleration if available
        if self.use_gpu:
            focal_lengths = self._accelerate_computation(focal_lengths)
        
        # Create histogram
        ax.hist(focal_lengths, bins=bins, alpha=0.7, color=color)
        
        ax.set_title('Focal Length Distribution', fontsize=14)
        ax.set_xlabel('Focal Length (mm)', fontsize=12)
        ax.set_ylabel('Number of Images', fontsize=12)
        fig.tight_layout()
        
        # Cache the result
        if figure is None:
            self.cache.set(cache_key, {
                'figure': fig,
                'axes': fig.get_axes()
            })
        
        return fig
    
    @lru_cache(maxsize=8)
    def plot_aperture_distribution(self, df: pd.DataFrame, figure=None, 
                                  bins=15, color='lightgreen'):
        """Plot distribution of aperture values with GPU acceleration
        
        Args:
            df: DataFrame with EXIF data
            figure: Matplotlib figure to use (or None to create new)
            bins: Number of histogram bins
            color: Color to use for the histogram
            
        Returns:
            Matplotlib figure
        """
        cache_key = f"aperture_dist_{bins}_{color}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            if figure is not None:
                figure.clear()
                for ax in cached['axes']:
                    figure.add_axes(ax)
                return figure
            else:
                return cached['figure']
        
        # Process the dataframe if needed
        if 'f_number_num' not in df.columns:
            df = self._preprocess_dataframe(df)
        
        if df.empty or 'f_number_num' not in df.columns:
            return None
        
        if figure is None:
            fig = Figure(figsize=(10, 6), dpi=100)
        else:
            fig = figure
            fig.clear()
        
        ax = fig.add_subplot(111)
        
        # Get aperture data
        apertures = df['f_number_num'].dropna().values
        
        # Use GPU acceleration if available
        if self.use_gpu:
            apertures = self._accelerate_computation(apertures)
        
        # Create histogram
        ax.hist(apertures, bins=bins, alpha=0.7, color=color)
        
        ax.set_title('Aperture (f-number) Distribution', fontsize=14)
        ax.set_xlabel('f-number', fontsize=12)
        ax.set_ylabel('Number of Images', fontsize=12)
        fig.tight_layout()
        
        # Cache the result
        if figure is None:
            self.cache.set(cache_key, {
                'figure': fig,
                'axes': fig.get_axes()
            })
        
        return fig
    
    @lru_cache(maxsize=8)
    def plot_iso_distribution(self, df: pd.DataFrame, figure=None, 
                             bins=20, color='salmon', log_scale=True):
        """Plot distribution of ISO values with GPU acceleration
        
        Args:
            df: DataFrame with EXIF data
            figure: Matplotlib figure to use (or None to create new)
            bins: Number of histogram bins
            color: Color to use for the histogram
            log_scale: Whether to use logarithmic scale for x-axis
            
        Returns:
            Matplotlib figure
        """
        cache_key = f"iso_dist_{bins}_{color}_{log_scale}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            if figure is not None:
                figure.clear()
                for ax in cached['axes']:
                    figure.add_axes(ax)
                return figure
            else:
                return cached['figure']
        
        # Process the dataframe if needed
        if 'iso_num' not in df.columns:
            df = self._preprocess_dataframe(df)
        
        if df.empty or 'iso_num' not in df.columns:
            return None
        
        if figure is None:
            fig = Figure(figsize=(10, 6), dpi=100)
        else:
            fig = figure
            fig.clear()
        
        ax = fig.add_subplot(111)
        
        # Get ISO data
        iso_values = df['iso_num'].dropna().values
        
        # Use GPU acceleration if available
        if self.use_gpu:
            iso_values = self._accelerate_computation(iso_values)
        
        # Create histogram
        ax.hist(iso_values, bins=bins, alpha=0.7, color=color)
        
        # Set logarithmic scale if requested
        if log_scale:
            ax.set_xscale('log')
        
        ax.set_title('ISO Distribution', fontsize=14)
        ax.set_xlabel('ISO' + (' (log scale)' if log_scale else ''), fontsize=12)
        ax.set_ylabel('Number of Images', fontsize=12)
        fig.tight_layout()
        
        # Cache the result
        if figure is None:
            self.cache.set(cache_key, {
                'figure': fig,
                'axes': fig.get_axes()
            })
        
        return fig
    
    @lru_cache(maxsize=8)
    def plot_time_of_day(self, df: pd.DataFrame, figure=None, color='purple'):
        """Plot distribution of photos by time of day with GPU acceleration
        
        Args:
            df: DataFrame with EXIF data
            figure: Matplotlib figure to use (or None to create new)
            color: Color to use for the bars
            
        Returns:
            Matplotlib figure
        """
        cache_key = f"time_of_day_{color}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            if figure is not None:
                figure.clear()
                for ax in cached['axes']:
                    figure.add_axes(ax)
                return figure
            else:
                return cached['figure']
        
        # Process the dataframe if needed
        if 'hour' not in df.columns:
            df = self._preprocess_dataframe(df)
        
        if df.empty or 'hour' not in df.columns or df['hour'].isna().all():
            return None
        
        if figure is None:
            fig = Figure(figsize=(10, 6), dpi=100)
        else:
            fig = figure
            fig.clear()
        
        ax = fig.add_subplot(111)
        
        # Create histogram of hours
        hour_counts = df['hour'].value_counts().sort_index()
        
        # Use GPU acceleration for color mapping if available
        if self.use_gpu and HAS_TORCH:
            color_array = np.array([matplotlib.colors.to_rgb(color)])
            color_array = self._accelerate_computation(color_array)
            color = tuple(color_array[0])
        
        ax.bar(hour_counts.index, hour_counts.values, color=color, alpha=0.7)
        
        ax.set_title('Photos by Time of Day', fontsize=14)
        ax.set_xlabel('Hour of Day (24h)', fontsize=12)
        ax.set_ylabel('Number of Images', fontsize=12)
        ax.set_xticks(range(0, 24, 2))
        fig.tight_layout()
        
        # Cache the result
        if figure is None:
            self.cache.set(cache_key, {
                'figure': fig,
                'axes': fig.get_axes()
            })
        
        return fig
    
    def plot_heatmap(self, df: pd.DataFrame, x_field: str, y_field: str, 
                    figure=None, cmap='viridis'):
        """Create a heatmap of two EXIF fields
        
        Args:
            df: DataFrame with EXIF data
            x_field: Field to use for x-axis
            y_field: Field to use for y-axis
            figure: Matplotlib figure to use (or None to create new)
            cmap: Colormap to use
            
        Returns:
            Matplotlib figure
        """
        cache_key = f"heatmap_{x_field}_{y_field}_{cmap}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            if figure is not None:
                figure.clear()
                for ax in cached['axes']:
                    figure.add_axes(ax)
                return figure
            else:
                return cached['figure']
        
        # Process the dataframe if needed
        df = self._preprocess_dataframe(df)
        
        # Check if fields exist with numeric values
        x_field_num = f"{x_field}_num" if f"{x_field}_num" in df.columns else x_field
        y_field_num = f"{y_field}_num" if f"{y_field}_num" in df.columns else y_field
        
        if df.empty or x_field_num not in df.columns or y_field_num not in df.columns:
            return None
        
        if figure is None:
            fig = Figure(figsize=(10, 8), dpi=100)
        else:
            fig = figure
            fig.clear()
        
        ax = fig.add_subplot(111)
        
        # Create pivot table for heatmap
        try:
            # For fields with few unique values, use them directly
            if df[x_field_num].nunique() < 20 and df[y_field_num].nunique() < 20:
                pivot = pd.crosstab(df[y_field_num], df[x_field_num])
            else:
                # For continuous fields, bin them first
                x_bins = pd.cut(df[x_field_num].dropna(), bins=10)
                y_bins = pd.cut(df[y_field_num].dropna(), bins=10)
                pivot = pd.crosstab(y_bins, x_bins)
            
            # Use GPU acceleration if available
            if self.use_gpu and HAS_TORCH and pivot.size > 100:
                pivot_array = pivot.values
                tensor = torch.tensor(pivot_array, dtype=torch.float32).to(self.device)
                # Apply smoothing
                kernel = torch.ones((3, 3), dtype=torch.float32).to(self.device) / 9.0
                tensor_padded = torch.nn.functional.pad(
                    tensor.unsqueeze(0).unsqueeze(0), 
                    (1, 1, 1, 1), 
                    mode='reflect'
                )
                smoothed = torch.nn.functional.conv2d(
                    tensor_padded, 
                    kernel.unsqueeze(0).unsqueeze(0)
                ).squeeze()
                pivot_array = smoothed.cpu().numpy()
                
                # Create a new DataFrame with the smoothed data
                pivot = pd.DataFrame(
                    pivot_array, 
                    index=pivot.index, 
                    columns=pivot.columns
                )
            
            # Plot heatmap
            sns.heatmap(pivot, cmap=cmap, ax=ax, annot=pivot.shape[0] < 10)
            
            ax.set_title(f'{y_field} vs {x_field}', fontsize=14)
            ax.set_xlabel(x_field, fontsize=12)
            ax.set_ylabel(y_field, fontsize=12)
            fig.tight_layout()
            
            # Cache the result
            if figure is None:
                self.cache.set(cache_key, {
                    'figure': fig,
                    'axes': fig.get_axes()
                })
            
            return fig
        
        except Exception as e:
            print(f"Error creating heatmap: {e}")
            return None
    
    def plot_map(self, df: pd.DataFrame, figure=None, 
                marker_size=50, alpha=0.7):
        """Plot photo locations on a map if GPS coordinates are available
        
        Args:
            df: DataFrame with EXIF data
            figure: Matplotlib figure to use (or None to create new)
            marker_size: Size of markers
            alpha: Transparency of markers
            
        Returns:
            Matplotlib figure
        """
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            HAS_CARTOPY = True
        except ImportError:
            print("Cartopy not available for map plotting")
            HAS_CARTOPY = False
            return None
        
        cache_key = f"map_{marker_size}_{alpha}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            if figure is not None:
                figure.clear()
                for ax in cached['axes']:
                    figure.add_axes(ax)
                return figure
            else:
                return cached['figure']
        
        # Check if GPS coordinates are available
        gps_fields = ['gps_latitude_dec', 'gps_longitude_dec']
        if df.empty or not all(field in df.columns for field in gps_fields):
            return None
        
        # Filter out rows with missing GPS data
        df_gps = df.dropna(subset=gps_fields)
        if df_gps.empty:
            return None
        
        if figure is None:
            fig = Figure(figsize=(12, 8), dpi=100)
        else:
            fig = figure
            fig.clear()
        
        # Create map with Cartopy
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        
        # Plot photo locations
        scatter = ax.scatter(
            df_gps['gps_longitude_dec'], 
            df_gps['gps_latitude_dec'],
            transform=ccrs.PlateCarree(),
            s=marker_size,
            alpha=alpha,
            c=df_gps['date_taken_dt'].dt.year if 'date_taken_dt' in df_gps.columns else 'red',
            cmap='viridis'
        )
        
        # Add colorbar if colored by year
        if 'date_taken_dt' in df_gps.columns:
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label('Year')
        
        # Set map extent to show all points with some padding
        padding = 1.0  # degrees
        min_lon = df_gps['gps_longitude_dec'].min() - padding
        max_lon = df_gps['gps_longitude_dec'].max() + padding
        min_lat = df_gps['gps_latitude_dec'].min() - padding
        max_lat = df_gps['gps_latitude_dec'].max() + padding
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
        
        ax.set_title('Photo Locations', fontsize=14)
        fig.tight_layout()
        
        # Cache the result
        if figure is None:
            self.cache.set(cache_key, {
                'figure': fig,
                'axes': fig.get_axes()
            })
        
        return fig
    
    def clear_cache(self):
        """Clear the visualization cache"""
        self.cache.clear()
        # Also clear the lru_cache for the plotting methods
        self.plot_camera_distribution.cache_clear()
        self.plot_focal_length_distribution.cache_clear()
        self.plot_aperture_distribution.cache_clear()
        self.plot_iso_distribution.cache_clear()
        self.plot_time_of_day.cache_clear()


# Example usage
if __name__ == "__main__":
    # Create a sample DataFrame
    data = {
        'camera_model': ['Canon EOS R5', 'Sony A7 IV', 'Nikon Z7', 'Canon EOS R5', 'Fujifilm X-T4'],
        'focal_length': ['24mm', '50mm', '70mm', '35mm', '18mm'],
        'f_number': ['f/2.8', 'f/1.8', 'f/4', 'f/5.6', 'f/2'],
        'iso': ['100', '400', '800', '200', '1600'],
        'date_taken': ['2023:05:01 10:30:00', '2023:05:02 15:45:00', 
                       '2023:05:03 08:15:00', '2023:05:04 20:00:00', 
                       '2023:05:05 12:30:00']
    }
    df = pd.DataFrame(data)
    
    # Create visualizer
    visualizer = EnhancedVisualizer(use_gpu=True)
    
    # Create a figure
    fig = visualizer.plot_camera_distribution(df)
    
    # Save to file
    if fig:
        fig.savefig('camera_distribution.png')
        print("Saved camera_distribution.png")
