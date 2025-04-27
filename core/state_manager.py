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
