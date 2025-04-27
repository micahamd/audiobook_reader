"""
Speech-to-Text engine module for the Audiobook Reader application.
Uses OpenAI's Whisper model for transcription.
"""

import os
from typing import Dict, Optional, Union

import whisper


class STTEngine:
    """Interface for the Speech-to-Text model (Whisper)."""
    
    AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large"]
    
    def __init__(self, model_name: str = "base", model_dir: Optional[str] = None):
        """
        Initialize the STT engine.
        
        Args:
            model_name: Name of the Whisper model to use.
            model_dir: Directory to store the model. If None, uses the default.
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model name: {model_name}. "
                             f"Available models: {', '.join(self.AVAILABLE_MODELS)}")
        
        self.model_name = model_name
        self.model_dir = model_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'whisper')
        self.model = None
    
    def load_model(self):
        """Load the Whisper model."""
        if self.model is None:
            # Create model directory if it doesn't exist
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Load the model
            self.model = whisper.load_model(self.model_name, download_root=self.model_dir)
    
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Union[str, float]]:
        """
        Transcribe audio to text.
        
        Args:
            audio_path: Path to the audio file.
            language: Optional language code. If None, Whisper will auto-detect.
            
        Returns:
            Dictionary with transcription results, including:
                - text: The transcribed text
                - segments: List of segments with timing information
        """
        if self.model is None:
            self.load_model()
        
        options = {}
        if language:
            options["language"] = language
        
        result = self.model.transcribe(audio_path, **options)
        
        # Convert the result to Markdown format
        markdown_text = result["text"]
        
        # Return the result with the markdown text
        return {
            "text": markdown_text,
            "segments": result["segments"]
        }
    
    def unload_model(self):
        """Unload the model to free memory."""
        self.model = None
