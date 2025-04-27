"""
Thread utilities for the Audiobook Reader application.
"""

from typing import Any, Callable, Dict, Optional

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal, pyqtSlot


class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.
    """
    started = pyqtSignal()
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    """
    Worker thread for running background tasks.
    """
    
    def __init__(self, fn: Callable, *args, **kwargs):
        """
        Initialize the worker.
        
        Args:
            fn: The function to run.
            *args: Arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.
        """
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        
        # Add the signals to the kwargs if they don't exist
        if 'progress_callback' not in kwargs:
            self.kwargs['progress_callback'] = self.signals.progress
    
    @pyqtSlot()
    def run(self):
        """
        Run the worker function with the provided arguments.
        """
        self.signals.started.emit()
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.result.emit(result)
        except Exception as e:
            self.signals.error.emit((e, None, None))
        finally:
            self.signals.finished.emit()


class ThreadManager:
    """
    Manages worker threads.
    """
    
    def __init__(self):
        """Initialize the thread manager."""
        self.threadpool = QThreadPool()
        print(f"Multithreading with maximum {self.threadpool.maxThreadCount()} threads")
    
    def start_worker(self, fn: Callable, *args, **kwargs) -> Worker:
        """
        Start a worker thread.
        
        Args:
            fn: The function to run.
            *args: Arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.
            
        Returns:
            The worker instance.
        """
        worker = Worker(fn, *args, **kwargs)
        self.threadpool.start(worker)
        return worker
