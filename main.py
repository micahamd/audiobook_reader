"""
Audiobook Reader application entry point.
"""

import sys
import os

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon

from ui.main_window import MainWindow


def main():
    """Main entry point for the application."""
    # Create the application
    app = QApplication(sys.argv)
    app.setApplicationName("Audiobook Reader")

    # Set application icon if available
    icon_path = os.path.join(os.path.dirname(__file__), 'resources', 'icons', 'app_icon.svg')
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # Create and show the main window
    window = MainWindow()
    window.show()

    # Run the application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
