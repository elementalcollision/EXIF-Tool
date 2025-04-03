#!/usr/bin/env python3
"""
Optimization utilities for camera extractors
Provides thread pooling, memory management, and performance monitoring
"""

import os
import gc
import time
import psutil
import threading
import multiprocessing
import concurrent.futures
from typing import Dict, Any, List, Callable, TypeVar, Generic, Optional, Tuple
import numpy as np
from functools import wraps
import logging
import traceback
import resource

# Type variable for generic functions
T = TypeVar('T')
R = TypeVar('R')

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('extractor_optimization')

class MemoryTracker:
    """Track memory usage during operations"""
    
    def __init__(self, memory_limit_fraction: float = 0.75):
        """Initialize memory tracker
        
        Args:
            memory_limit_fraction: Maximum memory usage as fraction of total memory
        """
        self.memory_limit_fraction = memory_limit_fraction
        self.total_memory = psutil.virtual_memory().total
        self.memory_limit = self.total_memory * memory_limit_fraction
        self.peak_memory = 0
        self.start_memory = 0
        
    def start(self):
        """Start tracking memory usage"""
        gc.collect()  # Force garbage collection before starting
        self.start_memory = psutil.Process(os.getpid()).memory_info().rss
        self.peak_memory = self.start_memory
        return self.start_memory
    
    def check(self) -> Tuple[int, float]:
        """Check current memory usage
        
        Returns:
            Tuple of (current memory in bytes, usage percentage of limit)
        """
        current = psutil.Process(os.getpid()).memory_info().rss
        if current > self.peak_memory:
            self.peak_memory = current
        
        percentage = (current / self.memory_limit) * 100
        return current, percentage
    
    def end(self) -> Tuple[int, int, float]:
        """End tracking and return statistics
        
        Returns:
            Tuple of (memory used, peak memory, peak percentage of limit)
        """
        gc.collect()  # Force garbage collection before measuring
        current = psutil.Process(os.getpid()).memory_info().rss
        memory_used = current - self.start_memory
        peak_percentage = (self.peak_memory / self.memory_limit) * 100
        
        return memory_used, self.peak_memory, peak_percentage
    
    def is_over_limit(self) -> bool:
        """Check if memory usage is over the limit
        
        Returns:
            True if over limit, False otherwise
        """
        current, percentage = self.check()
        return percentage > 100.0
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.end()
        if exc_type is not None:
            logger.error(f"Exception during memory-tracked operation: {exc_val}")
            return False
        return True


class ThreadPoolManager:
    """Manage thread pools for parallel processing"""
    
    def __init__(self, max_workers: Optional[int] = None):
        """Initialize thread pool manager
        
        Args:
            max_workers: Maximum number of worker threads (None = auto)
        """
        if max_workers is None:
            # Use n-2 cores on Apple Silicon, n-1 on other platforms
            import platform
            total_cores = multiprocessing.cpu_count()
            if platform.processor() == 'arm':
                self.max_workers = max(1, total_cores - 2)
            else:
                self.max_workers = max(1, total_cores - 1)
        else:
            self.max_workers = max_workers
            
        self.active_pools = []
    
    def get_pool(self) -> concurrent.futures.ThreadPoolExecutor:
        """Get a thread pool executor
        
        Returns:
            ThreadPoolExecutor instance
        """
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self.active_pools.append(pool)
        return pool
    
    def shutdown_all(self):
        """Shutdown all active thread pools"""
        for pool in self.active_pools:
            pool.shutdown(wait=True)
        self.active_pools = []
    
    def __enter__(self):
        """Context manager entry"""
        return self.get_pool()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown_all()
        if exc_type is not None:
            logger.error(f"Exception during thread pool operation: {exc_val}")
            return False
        return True


def parallel_map(func: Callable[[T], R], items: List[T], 
                max_workers: Optional[int] = None) -> List[R]:
    """Execute a function on multiple items in parallel
    
    Args:
        func: Function to execute
        items: List of items to process
        max_workers: Maximum number of worker threads
        
    Returns:
        List of results
    """
    if not items:
        return []
    
    with ThreadPoolManager(max_workers) as executor:
        # Submit all tasks
        future_to_item = {executor.submit(func, item): item for item in items}
        
        # Collect results as they complete
        results = []
        for future in concurrent.futures.as_completed(future_to_item):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                item = future_to_item[future]
                logger.error(f"Error processing {item}: {exc}")
                traceback.print_exc()
    
    return results


def chunked_processing(data: np.ndarray, func: Callable[[np.ndarray], Any], 
                      chunk_size: int = 1024*1024, 
                      memory_limit_fraction: float = 0.75) -> List[Any]:
    """Process large arrays in chunks to limit memory usage
    
    Args:
        data: NumPy array to process
        func: Function to apply to each chunk
        chunk_size: Size of each chunk in elements
        memory_limit_fraction: Maximum memory usage as fraction of total
        
    Returns:
        List of results from each chunk
    """
    if data.size <= chunk_size:
        return [func(data)]
    
    # Calculate number of chunks
    num_chunks = (data.size + chunk_size - 1) // chunk_size
    
    # Process each chunk
    results = []
    memory_tracker = MemoryTracker(memory_limit_fraction)
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, data.size)
        
        # Check memory before processing
        if memory_tracker.is_over_limit():
            logger.warning(f"Memory limit reached at chunk {i}/{num_chunks}. Forcing garbage collection.")
            gc.collect()
            
            # If still over limit, reduce chunk size
            if memory_tracker.is_over_limit():
                chunk_size = chunk_size // 2
                logger.warning(f"Reducing chunk size to {chunk_size}")
                
                # Recalculate number of chunks
                num_chunks = (data.size - start_idx + chunk_size - 1) // chunk_size
        
        # Process chunk
        chunk = data[start_idx:end_idx]
        try:
            result = func(chunk)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing chunk {i}: {e}")
            traceback.print_exc()
    
    return results


def set_memory_limit(memory_limit_fraction: float = 0.75):
    """Set memory limit for the current process
    
    Args:
        memory_limit_fraction: Maximum memory usage as fraction of total
    """
    # Get total system memory
    total_memory = psutil.virtual_memory().total
    memory_limit = int(total_memory * memory_limit_fraction)
    
    # Get current resource limits
    current_soft, current_hard = resource.getrlimit(resource.RLIMIT_AS)
    
    # Check if we're trying to increase beyond the hard limit
    if memory_limit > current_hard and current_hard != resource.RLIM_INFINITY:
        # Can't set limit higher than current hard limit
        logger.info(f"Using existing memory limit: {current_hard / (1024**3):.2f} GB")
        return current_hard
    
    # Set memory limit (soft and hard)
    try:
        # Try to set both soft and hard limits
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        logger.info(f"Memory limit set to {memory_limit / (1024**3):.2f} GB "
                   f"({memory_limit_fraction*100:.0f}% of total)")
    except (ValueError, resource.error) as e:
        try:
            # If that fails, try to set just the soft limit
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, current_hard))
            logger.info(f"Soft memory limit set to {memory_limit / (1024**3):.2f} GB")
        except (ValueError, resource.error):
            # If we can't set limits at all, just log it
            logger.info(f"Using soft memory monitoring with target: {memory_limit / (1024**3):.2f} GB")
            # We'll rely on the MemoryTracker class to monitor usage
    
    return memory_limit


def performance_monitor(func):
    """Decorator to monitor function performance
    
    Args:
        func: Function to monitor
    
    Returns:
        Wrapped function with performance monitoring
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss
        
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            end_time = time.time()
            logger.error(f"Error in {func.__name__}: {e}")
            logger.error(f"Execution time: {end_time - start_time:.2f} seconds")
            raise
        
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss
        memory_used = end_memory - start_memory
        
        logger.info(f"Function {func.__name__} completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Memory used: {memory_used / (1024**2):.2f} MB")
        
        return result
    
    return wrapper


class SafeArrayOperationContext:
    """Context manager for safe array operations
    
    This class provides a context manager interface for safe array operations,
    which can be used with the 'with' statement.
    
    Example:
        with SafeArrayOperationContext():
            # Code that might cause memory issues
            result = process_large_array(data)
    """
    
    def __init__(self):
        self.original_memory = None
    
    def __enter__(self):
        # Record memory usage on entry
        self.original_memory = psutil.Process(os.getpid()).memory_info().rss
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # If we had a memory error, try to recover
        if exc_type is MemoryError:
            # Force garbage collection
            gc.collect()
            logger.warning(f"Memory error in context, recovered with GC")
            return True  # Suppress the exception
        
        # Check if memory usage increased significantly
        current_memory = psutil.Process(os.getpid()).memory_info().rss
        if current_memory > self.original_memory * 2:  # More than doubled
            logger.warning(f"High memory usage detected: {(current_memory - self.original_memory) / (1024*1024):.2f} MB increase")
            gc.collect()  # Force garbage collection
        
        return False  # Don't suppress other exceptions


def safe_array_operation(func=None):
    """Make array operations memory-safe
    
    This function can be used in two ways:
    1. As a decorator: @safe_array_operation
    2. As a context manager: with safe_array_operation():
    
    Args:
        func: Function to make memory-safe (when used as decorator)
    
    Returns:
        Wrapped function with memory safety or context manager
    """
    # If used as context manager (with safe_array_operation():)
    if func is None:
        return SafeArrayOperationContext()
    
    # If used as decorator (@safe_array_operation)
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Try with original arguments
            return func(*args, **kwargs)
        except MemoryError:
            # Force garbage collection
            gc.collect()
            
            # Try again with reduced memory usage
            try:
                logger.warning(f"Memory error in {func.__name__}, retrying with reduced memory")
                
                # If first argument is a numpy array, try to reduce its size
                if len(args) > 0 and isinstance(args[0], np.ndarray):
                    array = args[0]
                    
                    # If array is too large, process in chunks
                    if array.size > 1024*1024:  # 1M elements
                        logger.info(f"Processing large array in chunks")
                        
                        # Create new args with smaller first argument
                        new_args = list(args)
                        
                        # Process in chunks
                        results = []
                        chunk_size = array.shape[0] // 4  # Start with 1/4 of the array
                        
                        for i in range(0, array.shape[0], chunk_size):
                            end = min(i + chunk_size, array.shape[0])
                            chunk = array[i:end]
                            
                            # Replace first argument with chunk
                            new_args[0] = chunk
                            
                            # Call function with chunk
                            chunk_result = func(*new_args, **kwargs)
                            results.append(chunk_result)
                        
                        # Combine results
                        if isinstance(results[0], np.ndarray):
                            return np.concatenate(results)
                        elif isinstance(results[0], dict):
                            combined = {}
                            for r in results:
                                combined.update(r)
                            return combined
                        else:
                            return results
                
                # If we can't optimize, just try again
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__} even after memory optimization: {e}")
                raise
    
    return wrapper
