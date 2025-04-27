"""
Helper functions for the Audiobook Reader application.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple


def validate_file_path(file_path: str) -> bool:
    """
    Validate that a file path exists and is readable.
    
    Args:
        file_path: The file path to validate.
        
    Returns:
        True if the file exists and is readable, False otherwise.
    """
    try:
        return os.path.isfile(file_path) and os.access(file_path, os.R_OK)
    except (TypeError, ValueError):
        return False


def get_supported_audio_extensions() -> List[str]:
    """
    Get a list of supported audio file extensions.
    
    Returns:
        List of extensions with dots (e.g., ['.mp3', '.wav']).
    """
    return ['.mp3', '.wav']


def get_supported_text_extensions() -> List[str]:
    """
    Get a list of supported text file extensions.
    
    Returns:
        List of extensions with dots.
    """
    # This is a simplified list - markitdown supports many more formats
    return ['.txt', '.md', '.markdown', '.docx', '.doc', '.pdf', '.rtf', '.odt']


def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string (MM:SS).
    
    Args:
        seconds: Time in seconds.
        
    Returns:
        Formatted time string.
    """
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def split_text_into_chunks(text: str, max_length: int = 1000) -> List[str]:
    """
    Split text into chunks of maximum length.
    
    Args:
        text: The text to split.
        max_length: Maximum length of each chunk.
        
    Returns:
        List of text chunks.
    """
    # Simple implementation - split by paragraphs first
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed max_length, start a new chunk
        if len(current_chunk) + len(paragraph) > max_length and current_chunk:
            chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += '\n\n' + paragraph
            else:
                current_chunk = paragraph
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks
