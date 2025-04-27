"""
Text processing module for the Audiobook Reader application.
Handles loading and converting various file formats to Markdown using markitdown.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import markitdown


class TextProcessor:
    """Handles text file loading and conversion to Markdown."""
    
    def __init__(self, temp_dir: str = None):
        """
        Initialize the text processor.
        
        Args:
            temp_dir: Directory for temporary files. If None, uses the default temp directory.
        """
        self.temp_dir = temp_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'temp')
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def load_file(self, file_path: str) -> Tuple[str, str]:
        """
        Load a file and convert it to Markdown if necessary.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            Tuple of (markdown_content, original_format)
            
        Raises:
            ValueError: If the file cannot be processed.
        """
        file_path = Path(file_path)
        file_format = file_path.suffix.lower().lstrip('.')
        
        # If it's already a markdown file, just read it
        if file_format in ['md', 'markdown']:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content, file_format
        
        # Otherwise, use markitdown to convert
        try:
            markdown_content = markitdown.markitdown(file_path)
            return markdown_content, file_format
        except Exception as e:
            raise ValueError(f"Failed to convert file to Markdown: {e}")
    
    def save_markdown(self, content: str, original_path: Optional[str] = None) -> str:
        """
        Save Markdown content to a temporary file.
        
        Args:
            content: The Markdown content to save.
            original_path: Optional original file path to derive the filename.
            
        Returns:
            Path to the saved Markdown file.
        """
        if original_path:
            base_name = Path(original_path).stem
            temp_path = os.path.join(self.temp_dir, f"{base_name}.md")
        else:
            temp_path = os.path.join(self.temp_dir, "converted.md")
        
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return temp_path
