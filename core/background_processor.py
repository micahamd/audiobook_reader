"""
Background processor for pre-processing upcoming pages.
"""

import os
import time
import threading
import queue
from typing import Dict, List, Callable, Any, Optional

import tempfile


class BackgroundProcessor:
    """
    A background processor that pre-processes upcoming pages
    and caches the results for instant access.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the background processor.
        
        Args:
            cache_dir: Directory for caching processed results.
                      If None, a temporary directory will be used.
        """
        # Set up cache directory
        if cache_dir:
            self.cache_dir = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
        else:
            self.cache_dir = os.path.join(tempfile.gettempdir(), 'audiobook_reader_cache')
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Task queue for background processing
        self.task_queue = queue.PriorityQueue()
        
        # Results cache
        self.results_cache = {}
        
        # Worker thread
        self.worker_thread = None
        self.stop_requested = False
        
        # Start the worker thread
        self.start_worker()
    
    def start_worker(self):
        """Start the worker thread."""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.stop_requested = False
            self.worker_thread = threading.Thread(target=self._worker_loop)
            self.worker_thread.daemon = True
            self.worker_thread.start()
    
    def stop_worker(self):
        """Stop the worker thread."""
        self.stop_requested = True
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
    
    def _worker_loop(self):
        """Worker thread loop."""
        while not self.stop_requested:
            try:
                # Get the next task from the queue with a timeout
                priority, task_id, processor_func, args, kwargs, callback = self.task_queue.get(timeout=0.5)
                
                # Process the task
                try:
                    print(f"Processing task {task_id} with priority {priority}")
                    result = processor_func(*args, **kwargs)
                    
                    # Cache the result
                    self.results_cache[task_id] = result
                    
                    # Call the callback if provided
                    if callback:
                        callback(task_id, result)
                        
                except Exception as e:
                    print(f"Error processing task {task_id}: {str(e)}")
                
                # Mark the task as done
                self.task_queue.task_done()
                
            except queue.Empty:
                # No tasks in the queue, sleep a bit
                time.sleep(0.1)
    
    def add_task(self, task_id: str, processor_func: Callable, priority: int = 0, 
                 callback: Optional[Callable] = None, *args, **kwargs):
        """
        Add a task to the processing queue.
        
        Args:
            task_id: Unique identifier for the task.
            processor_func: Function to process the task.
            priority: Priority of the task (lower values = higher priority).
            callback: Function to call when the task is complete.
            *args: Arguments to pass to the processor function.
            **kwargs: Keyword arguments to pass to the processor function.
        """
        # Check if the task is already in the cache
        if task_id in self.results_cache:
            # Task already processed, call the callback immediately
            if callback:
                callback(task_id, self.results_cache[task_id])
            return
        
        # Add the task to the queue
        self.task_queue.put((priority, task_id, processor_func, args, kwargs, callback))
    
    def get_result(self, task_id: str) -> Any:
        """
        Get the result of a processed task.
        
        Args:
            task_id: The task identifier.
            
        Returns:
            The result of the task, or None if not available.
        """
        return self.results_cache.get(task_id)
    
    def clear_cache(self, task_id: Optional[str] = None):
        """
        Clear the cache.
        
        Args:
            task_id: If provided, only clear this task from the cache.
                   If None, clear the entire cache.
        """
        if task_id:
            if task_id in self.results_cache:
                del self.results_cache[task_id]
        else:
            self.results_cache.clear()
    
    def clear_cache_files(self):
        """Clear all cache files from the cache directory."""
        for filename in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting cache file {file_path}: {str(e)}")
    
    def __del__(self):
        """Clean up resources."""
        self.stop_worker()
