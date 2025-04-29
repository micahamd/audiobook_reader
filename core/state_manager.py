"""
State manager module for the Audiobook Reader application.
Handles loading and saving application state.
"""

import json
import os
from typing import Any, Dict, Optional


class StateManager:
    """Manages loading/saving application state."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the state manager.

        Args:
            config_path: Path to the config file. If None, uses the default.
        """
        self.config_path = config_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')

        # Create a directory for storing edited files
        self.edited_files_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'edited_files')
        os.makedirs(self.edited_files_dir, exist_ok=True)

        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """
        Load state from the config file.

        Returns:
            Dictionary with the state.
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                # If the file is corrupted or can't be read, return default state
                return self._get_default_state()
        else:
            return self._get_default_state()

    def _get_default_state(self) -> Dict[str, Any]:
        """
        Get the default state.

        Returns:
            Dictionary with the default state.
        """
        return {
            "last_file": None,
            "last_position": 0,
            "edited_files": {},  # Map of original file path to edited file path
            "tts_settings": {
                "voice": "default",
                "speed": 1.0
            },
            "stt_settings": {
                "model": "base",
                "language": None  # Auto-detect
            },
            "window_size": [800, 600],
            "window_position": [100, 100]
        }

    def save_state(self):
        """Save the current state to the config file."""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2)
        except IOError:
            # Log error but don't crash
            print(f"Error: Could not save state to {self.config_path}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the state.

        Args:
            key: The key to get.
            default: Default value if the key doesn't exist.

        Returns:
            The value or default.
        """
        return self.state.get(key, default)

    def set(self, key: str, value: Any):
        """
        Set a value in the state.

        Args:
            key: The key to set.
            value: The value to set.
        """
        self.state[key] = value

    def update(self, updates: Dict[str, Any]):
        """
        Update multiple values in the state.

        Args:
            updates: Dictionary with updates.
        """
        self.state.update(updates)

    def get_edited_file_path(self, original_file_path: str) -> Optional[str]:
        """
        Get the path to the edited version of a file.

        Args:
            original_file_path: Path to the original file.

        Returns:
            Path to the edited file, or None if no edited version exists.
        """
        edited_files = self.state.get("edited_files", {})
        return edited_files.get(original_file_path)

    def has_edited_version(self, original_file_path: str) -> bool:
        """
        Check if a file has an edited version.

        Args:
            original_file_path: Path to the original file.

        Returns:
            True if an edited version exists, False otherwise.
        """
        edited_file_path = self.get_edited_file_path(original_file_path)
        return edited_file_path is not None and os.path.exists(edited_file_path)

    def save_edited_file(self, original_file_path: str, content: str) -> str:
        """
        Save an edited version of a file.

        Args:
            original_file_path: Path to the original file.
            content: The edited content to save.

        Returns:
            Path to the saved edited file.
        """
        # Create a filename for the edited version
        original_filename = os.path.basename(original_file_path)
        base_name, ext = os.path.splitext(original_filename)
        edited_filename = f"{base_name}_edited{ext}"
        edited_file_path = os.path.join(self.edited_files_dir, edited_filename)

        # Save the content
        try:
            with open(edited_file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # Update the state
            edited_files = self.state.get("edited_files", {})
            edited_files[original_file_path] = edited_file_path
            self.state["edited_files"] = edited_files

            # Save the state
            self.save_state()

            return edited_file_path
        except Exception as e:
            print(f"Error saving edited file: {str(e)}")
            return original_file_path  # Fall back to the original file

    def delete_edited_file(self, original_file_path: str) -> bool:
        """
        Delete an edited version of a file.

        Args:
            original_file_path: Path to the original file.

        Returns:
            True if the file was deleted, False otherwise.
        """
        edited_file_path = self.get_edited_file_path(original_file_path)
        if edited_file_path and os.path.exists(edited_file_path):
            try:
                os.remove(edited_file_path)

                # Update the state
                edited_files = self.state.get("edited_files", {})
                if original_file_path in edited_files:
                    del edited_files[original_file_path]

                # Save the state
                self.save_state()

                return True
            except Exception as e:
                print(f"Error deleting edited file: {str(e)}")
                return False
        return False
