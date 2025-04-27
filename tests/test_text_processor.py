"""
Tests for the text processor module.
"""

import os
import unittest
from pathlib import Path
import tempfile

from core.text_processor import TextProcessor


class TestTextProcessor(unittest.TestCase):
    """Tests for the TextProcessor class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.text_processor = TextProcessor()
        self.temp_dir = tempfile.TemporaryDirectory()
    
    def tearDown(self):
        """Clean up the test environment."""
        self.temp_dir.cleanup()
    
    def test_load_markdown_file(self):
        """Test loading a Markdown file."""
        # Create a test Markdown file
        test_content = "# Test Heading\n\nThis is a test."
        test_file = Path(self.temp_dir.name) / "test.md"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        # Load the file
        content, format_str = self.text_processor.load_file(str(test_file))
        
        # Check the result
        self.assertEqual(content, test_content)
        self.assertEqual(format_str, "md")
    
    def test_save_markdown(self):
        """Test saving Markdown content."""
        # Test content
        test_content = "# Test Heading\n\nThis is a test."
        
        # Save the content
        saved_path = self.text_processor.save_markdown(test_content)
        
        # Check that the file exists
        self.assertTrue(os.path.exists(saved_path))
        
        # Check the content
        with open(saved_path, 'r', encoding='utf-8') as f:
            saved_content = f.read()
        
        self.assertEqual(saved_content, test_content)


if __name__ == '__main__':
    unittest.main()
