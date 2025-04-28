"""
Virtual text display widget for efficient handling of large texts.
"""

from PyQt6.QtWidgets import QTextEdit
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QTextCursor


class VirtualTextDisplay(QTextEdit):
    """
    A text display widget that efficiently handles large texts.
    This is a simplified version that focuses on stability over virtual scrolling.
    """
    # Signal emitted when the content changes
    content_changed = pyqtSignal()
    # Signal emitted when the user edits the text
    text_edited = pyqtSignal()

    def __init__(self, parent=None):
        """Initialize the virtual text display."""
        super().__init__(parent)

        # Full text content
        self.full_text = ""

        # Current page information
        self.current_page_text = ""
        self.current_page_start = 0
        self.current_page_end = 0

        # Connect signals
        self.textChanged.connect(self.handle_text_change)

        # Initialize with empty content
        self.setReadOnly(True)

    def set_full_text(self, text):
        """
        Set the full text content.

        Args:
            text: The complete text content.
        """
        self.full_text = text
        self.current_page_text = text
        self.current_page_start = 0
        self.current_page_end = len(text)
        self.setPlainText(text)

    def get_full_text(self):
        """
        Get the full text content.

        Returns:
            The complete text content.
        """
        # If the text has been edited, update the full text
        if not self.isReadOnly():
            # Get the current text
            current_text = self.toPlainText()

            # Update the full text with the edited content
            if self.current_page_start == 0 and self.current_page_end >= len(self.full_text):
                # We're showing the entire text
                self.full_text = current_text
            else:
                # We're showing a portion of the text
                prefix = self.full_text[:self.current_page_start]
                suffix = self.full_text[self.current_page_end:]
                self.full_text = prefix + current_text + suffix

        return self.full_text

    def handle_text_change(self):
        """Handle text changes."""
        if not self.isReadOnly():
            self.text_edited.emit()

    def set_page_text(self, text, page_start=0, page_end=None):
        """
        Set the text for the current page.

        Args:
            text: The text to display.
            page_start: The start position of this page in the full text.
            page_end: The end position of this page in the full text.
        """
        if page_end is None:
            page_end = page_start + len(text)

        # Save the current cursor position relative to the page
        cursor = self.textCursor()
        relative_position = cursor.position()

        # Update the page information
        self.current_page_text = text
        self.current_page_start = page_start
        self.current_page_end = page_end

        # Update the displayed text
        self.setPlainText(text)

        # Try to restore the cursor position
        if relative_position <= len(text):
            cursor = self.textCursor()
            cursor.setPosition(min(relative_position, len(text)))
            self.setTextCursor(cursor)

        # Emit the content changed signal
        self.content_changed.emit()

    def get_absolute_cursor_position(self):
        """
        Get the absolute cursor position in the full text.

        Returns:
            The cursor position in the full text.
        """
        return self.current_page_start + self.textCursor().position()

    def set_absolute_cursor_position(self, position):
        """
        Set the cursor position in the full text.

        Args:
            position: The absolute position in the full text.
        """
        # Check if the position is within the current page
        if position >= self.current_page_start and position <= self.current_page_end:
            # Convert to page-relative position
            relative_position = position - self.current_page_start

            # Set the cursor position
            cursor = self.textCursor()
            cursor.setPosition(min(relative_position, len(self.current_page_text)))
            self.setTextCursor(cursor)
            self.ensureCursorVisible()

    def highlight_text(self, start, end):
        """
        Highlight text from start to end position.

        Args:
            start: Start position in the full text.
            end: End position in the full text.
        """
        # Check if the range is within the current page
        if start >= self.current_page_start and end <= self.current_page_end:
            # Convert to page-relative positions
            relative_start = start - self.current_page_start
            relative_end = end - self.current_page_start

            # Highlight the text
            cursor = self.textCursor()
            cursor.setPosition(relative_start)
            cursor.setPosition(relative_end, QTextCursor.MoveMode.KeepAnchor)
            self.setTextCursor(cursor)
            self.ensureCursorVisible()
