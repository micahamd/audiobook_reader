"""
Text processing module for the Audiobook Reader application.
Handles loading and converting various file formats to Markdown using markitdown.
All files are automatically converted to markdown and stored in a dedicated directory.
"""

import os
import hashlib
import time
import fnmatch
from pathlib import Path
from typing import Optional, Tuple, Dict

from markitdown import MarkItDown


class TextProcessor:
    """Handles text file loading and conversion to Markdown."""

    def __init__(self, markdown_dir: str = None, temp_dir: str = None):
        """
        Initialize the text processor.

        Args:
            markdown_dir: Directory for storing markdown files. If None, uses the default.
            temp_dir: Directory for temporary files. If None, uses the default temp directory.
        """
        self.temp_dir = temp_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'temp')
        os.makedirs(self.temp_dir, exist_ok=True)

        # Create a dedicated directory for markdown files
        self.markdown_dir = markdown_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'markdown_files')
        os.makedirs(self.markdown_dir, exist_ok=True)

        # Initialize MarkItDown
        self.md = MarkItDown(enable_plugins=False)

        # Cache of file paths to markdown paths
        self.file_path_cache: Dict[str, str] = {}

    def find_existing_markdown(self, file_path: str) -> Optional[str]:
        """
        Find an existing markdown file for the given original file.

        Args:
            file_path: Path to the original file.

        Returns:
            Path to the existing markdown file, or None if not found.
        """
        # Get the base name of the file
        original_filename = os.path.basename(file_path)
        base_name, _ = os.path.splitext(original_filename)

        # Look for files matching the pattern in the markdown directory
        pattern = f"{base_name}_*.md"
        matching_files = []

        for filename in os.listdir(self.markdown_dir):
            if fnmatch.fnmatch(filename, pattern):
                matching_files.append(os.path.join(self.markdown_dir, filename))

        if matching_files:
            # If we found matching files, use the most recently modified one
            most_recent = max(matching_files, key=os.path.getmtime)
            print(f"Found existing markdown file: {most_recent}")
            return most_recent

        return None

    def load_file(self, file_path: str) -> Tuple[str, str]:
        """
        Load a file and convert it to Markdown if necessary.
        Always saves a markdown version of the file in the markdown directory.

        Args:
            file_path: Path to the file.

        Returns:
            Tuple of (markdown_content, original_format)

        Raises:
            ValueError: If the file cannot be processed.
        """
        file_path = Path(file_path)
        file_format = file_path.suffix.lower().lstrip('.')

        # First, try to find an existing markdown file
        existing_markdown = self.find_existing_markdown(str(file_path))
        if existing_markdown:
            try:
                with open(existing_markdown, 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"Loaded existing markdown file: {existing_markdown}")

                # Update the cache to use this file
                self.file_path_cache[str(file_path)] = existing_markdown

                return content, 'md'
            except Exception as e:
                print(f"Error loading existing markdown file: {e}")
                # If there's an error, continue to create a new one

        # If no existing file was found or there was an error, we'll create a new markdown file

        # If it's already a markdown file, just read it and save a copy
        if file_format in ['md', 'markdown']:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Save a copy in our markdown directory
                self.save_markdown(content, str(file_path))
                return content, file_format
            except Exception as e:
                raise ValueError(f"Failed to read markdown file: {e}")

        # Otherwise, use markitdown to convert
        try:
            result = self.md.convert(str(file_path))
            content = result.text_content

            # Save the converted content to our markdown directory
            self.save_markdown(content, str(file_path))
            return content, file_format
        except Exception as e:
            raise ValueError(f"Failed to convert file to Markdown: {e}")

    def get_markdown_path(self, original_path: str) -> str:
        """
        Get the path where the markdown version of a file should be stored.

        Args:
            original_path: Path to the original file.

        Returns:
            Path where the markdown version should be stored.
        """
        # Check cache first
        if original_path in self.file_path_cache:
            return self.file_path_cache[original_path]

        # Create a filename based on the original path
        original_filename = os.path.basename(original_path)
        base_name, _ = os.path.splitext(original_filename)

        # Create a unique filename based on the filename only
        # Use just the filename for the hash to ensure consistency across sessions
        file_hash = hashlib.md5(original_filename.lower().encode()).hexdigest()[:8]
        markdown_filename = f"{base_name}_{file_hash}.md"
        markdown_path = os.path.join(self.markdown_dir, markdown_filename)

        # Cache the result
        self.file_path_cache[original_path] = markdown_path
        return markdown_path

    def save_markdown(self, content: str, original_path: Optional[str] = None) -> str:
        """
        Save Markdown content to the markdown directory.

        Args:
            content: The Markdown content to save.
            original_path: Original file path to derive the filename.

        Returns:
            Path to the saved Markdown file.
        """
        if not original_path:
            # If no original path, save to a temporary file
            temp_path = os.path.join(self.temp_dir, f"temp_{int(time.time())}.md")
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return temp_path

        # Get the path where this file should be stored
        markdown_path = self.get_markdown_path(original_path)

        # Save the content
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Saved markdown file: {markdown_path}")
        return markdown_path
