"""
Audio processing module for the Audiobook Reader application.
Handles loading and basic processing of audio files (MP3, WAV).
"""

import os
import tempfile
from pathlib import Path
from typing import Tuple, Optional

from pydub import AudioSegment


class AudioProcessor:
    """Handles audio file loading and processing."""
    
    SUPPORTED_FORMATS = ['mp3', 'wav']
    
    def __init__(self, temp_dir: str = None):
        """
        Initialize the audio processor.
        
        Args:
            temp_dir: Directory for temporary files. If None, uses system temp dir.
        """
        self.temp_dir = temp_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'temp')
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def load_audio(self, file_path: str) -> Tuple[AudioSegment, str]:
        """
        Load an audio file and return the AudioSegment and format.
        
        Args:
            file_path: Path to the audio file.
            
        Returns:
            Tuple of (AudioSegment, format_str)
            
        Raises:
            ValueError: If the file format is not supported.
        """
        file_path = Path(file_path)
        file_format = file_path.suffix.lower().lstrip('.')
        
        if file_format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported audio format: {file_format}. "
                             f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}")
        
        audio = AudioSegment.from_file(file_path, format=file_format)
        return audio, file_format
    
    def convert_to_wav(self, audio: AudioSegment) -> str:
        """
        Convert audio to WAV format and save to a temporary file.
        
        Args:
            audio: The AudioSegment to convert.
            
        Returns:
            Path to the temporary WAV file.
        """
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', dir=self.temp_dir, delete=False)
        temp_file.close()
        audio.export(temp_file.name, format='wav')
        return temp_file.name
    
    def get_audio_duration(self, audio: AudioSegment) -> float:
        """
        Get the duration of the audio in seconds.
        
        Args:
            audio: The AudioSegment.
            
        Returns:
            Duration in seconds.
        """
        return len(audio) / 1000.0  # pydub uses milliseconds
