"""
Bookmarks dialog for the Audiobook Reader application.
"""

import os

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QMessageBox, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon


class BookmarksDialog(QDialog):
    """Dialog for managing bookmarks."""

    # Signal emitted when a bookmark is selected
    bookmark_selected = pyqtSignal(dict)

    # Signal emitted when bookmarks are modified
    bookmarks_modified = pyqtSignal(list)

    def __init__(self, bookmarks, parent=None):
        """
        Initialize the bookmarks dialog.

        Args:
            bookmarks: List of bookmark dictionaries.
            parent: Parent widget.
        """
        super().__init__(parent)

        self.setWindowTitle("Bookmarks")
        self.setMinimumSize(500, 400)

        self.bookmarks = bookmarks or []

        # Load icons
        self.get_bookmark_icon = QIcon(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'icons', 'get_bookmark.svg'))
        self.add_bookmark_icon = QIcon(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'icons', 'add_bookmark.svg'))

        # Set dialog icon
        self.setWindowIcon(self.get_bookmark_icon)

        # Create layout
        layout = QVBoxLayout()

        # Add header
        header_label = QLabel("Your Bookmarks")
        header_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(header_label)

        # Add description
        description = QLabel("Select a bookmark to jump to that position in the text.")
        layout.addWidget(description)

        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

        # Create bookmarks list
        self.bookmarks_list = QListWidget()
        self.bookmarks_list.setAlternatingRowColors(True)
        self.bookmarks_list.itemDoubleClicked.connect(self.on_bookmark_double_clicked)
        layout.addWidget(self.bookmarks_list)

        # Populate the list
        self.populate_bookmarks_list()

        # Add buttons
        button_layout = QHBoxLayout()

        self.goto_button = QPushButton(self.get_bookmark_icon, "Go to Bookmark")
        self.goto_button.clicked.connect(self.on_goto_clicked)
        button_layout.addWidget(self.goto_button)

        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.on_remove_clicked)
        button_layout.addWidget(self.remove_button)

        self.remove_all_button = QPushButton("Remove All")
        self.remove_all_button.clicked.connect(self.on_remove_all_clicked)
        button_layout.addWidget(self.remove_all_button)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        button_layout.addWidget(close_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Update button states
        self.update_button_states()

    def populate_bookmarks_list(self):
        """Populate the bookmarks list."""
        self.bookmarks_list.clear()

        for i, bookmark in enumerate(self.bookmarks):
            # Format the bookmark information
            file_name = bookmark.get("file_path", "").split("/")[-1].split("\\")[-1]
            position_str = self.format_time(bookmark.get("position", 0))
            text_preview = bookmark.get("text", "")[:50] + "..." if len(bookmark.get("text", "")) > 50 else bookmark.get("text", "")

            # Create a formatted string
            item_text = f"Bookmark {i+1}: {file_name}\n"
            item_text += f"Position: {position_str}\n"
            item_text += f"Text: {text_preview}"

            # Create and add the item
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, bookmark)
            self.bookmarks_list.addItem(item)

    def format_time(self, seconds):
        """Format time in seconds to MM:SS format."""
        minutes = int(seconds / 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def update_button_states(self):
        """Update button states based on selection."""
        has_selection = self.bookmarks_list.currentItem() is not None
        has_bookmarks = self.bookmarks_list.count() > 0

        self.goto_button.setEnabled(has_selection)
        self.remove_button.setEnabled(has_selection)
        self.remove_all_button.setEnabled(has_bookmarks)

    def on_bookmark_double_clicked(self, item):
        """Handle bookmark double-click."""
        bookmark = item.data(Qt.ItemDataRole.UserRole)
        self.bookmark_selected.emit(bookmark)
        self.accept()

    def on_goto_clicked(self):
        """Handle goto button click."""
        current_item = self.bookmarks_list.currentItem()
        if current_item:
            bookmark = current_item.data(Qt.ItemDataRole.UserRole)
            self.bookmark_selected.emit(bookmark)
            self.accept()

    def on_remove_clicked(self):
        """Handle remove button click."""
        current_row = self.bookmarks_list.currentRow()
        if current_row >= 0:
            # Remove from the list widget
            self.bookmarks_list.takeItem(current_row)

            # Remove from the bookmarks list
            del self.bookmarks[current_row]

            # Emit the modified signal
            self.bookmarks_modified.emit(self.bookmarks)

            # Update button states
            self.update_button_states()

    def on_remove_all_clicked(self):
        """Handle remove all button click."""
        # Confirm with the user
        response = QMessageBox.question(
            self,
            "Remove All Bookmarks",
            "Are you sure you want to remove all bookmarks?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if response == QMessageBox.StandardButton.Yes:
            # Clear the list widget
            self.bookmarks_list.clear()

            # Clear the bookmarks list
            self.bookmarks = []

            # Emit the modified signal
            self.bookmarks_modified.emit(self.bookmarks)

            # Update button states
            self.update_button_states()
