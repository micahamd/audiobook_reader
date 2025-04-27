"""
Text-to-Speech engine module for the Audiobook Reader application.
Uses the Kokoro ONNX model for speech synthesis with timing information.
"""

import os
import tempfile
import warnings
import threading
import asyncio
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import soundfile as sf
import sounddevice as sd

# Try to import kokoro_onnx, but handle import errors gracefully
try:
    from kokoro_onnx import Kokoro
    KOKORO_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Failed to import kokoro_onnx: {str(e)}. TTS functionality will be limited.")
    KOKORO_AVAILABLE = False


class KokoroOnnxEngine:
    """Interface for the Kokoro ONNX TTS engine."""
    
    def __init__(self, model_path: Optional[str] = None, voices_path: Optional[str] = None, temp_dir: Optional[str] = None):
        """
        Initialize the TTS engine.
        
        Args:
            model_path: Path to the Kokoro model file. If None, uses the default.
            voices_path: Path to the voices file. If None, uses the default.
            temp_dir: Directory for temporary files. If None, uses the default.
        """
        # Set default paths
        self.model_path = model_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'kokoro', 'kokoro-v0_19.onnx')
        self.voices_path = voices_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'kokoro', 'voices.json')
        self.temp_dir = temp_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'temp')
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        self.kokoro = None
        self.current_text = ""
        self.word_timings = []
        self.stop_requested = False
        self.pause_requested = False
        self.audio_thread = None
    
    def load_model(self):
        """Load the Kokoro model."""
        if not KOKORO_AVAILABLE:
            warnings.warn("Cannot load model: kokoro_onnx is not available. "
                         "Please install the required dependencies.")
            return False
            
        if self.kokoro is None:
            # Check if model files exist
            if not os.path.exists(self.model_path):
                warnings.warn(f"Kokoro model file not found at {self.model_path}. "
                             f"Using fallback synthesis.")
                return False
            
            if not os.path.exists(self.voices_path):
                warnings.warn(f"Voices file not found at {self.voices_path}. "
                             f"Using fallback synthesis.")
                return False
            
            try:
                # Load the model
                self.kokoro = Kokoro(self.model_path, self.voices_path)
                print("Kokoro model loaded successfully")
                return True
            except Exception as e:
                warnings.warn(f"Failed to load Kokoro model: {str(e)}. Using fallback synthesis.")
                self.kokoro = None
                return False
        
        return True
    
    def synthesize(self, text: str, voice: str = "af_sarah", speed: float = 1.0) -> Tuple[str, List[Dict[str, Union[str, float]]]]:
        """
        Synthesize speech from text.
        
        Args:
            text: The text to synthesize.
            voice: The voice to use.
            speed: The speed factor (1.0 is normal speed).
            
        Returns:
            Tuple of (audio_path, word_timings)
            where word_timings is a list of dictionaries with:
                - word: The word
                - start: Start time in seconds
                - end: End time in seconds
        """
        # Store the current text
        self.current_text = text
        
        # Check if kokoro is available
        if not KOKORO_AVAILABLE:
            return self._dummy_synthesize(text, speed)
        
        # Load the model if not loaded
        if not self.load_model():
            return self._dummy_synthesize(text, speed)
        
        try:
            # Generate speech using Kokoro
            print(f"Synthesizing speech with voice: {voice}, speed: {speed}")
            
            # Create a temporary file for the audio
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', dir=self.temp_dir, delete=False)
            temp_file.close()
            
            # Generate speech
            samples, sample_rate = self.kokoro.create(
                text, voice=voice, speed=speed, lang="en-us"
            )
            
            # Save the audio to a file
            sf.write(temp_file.name, samples, sample_rate)
            
            # Calculate duration
            duration = len(samples) / sample_rate
            
            # Extract word timings
            word_timings = self._extract_word_timings(text.split(), duration)
            self.word_timings = word_timings
            
            return temp_file.name, word_timings
                
        except Exception as e:
            warnings.warn(f"Failed to synthesize speech: {str(e)}. Using fallback synthesis.")
            return self._dummy_synthesize(text, speed)
    
    def _dummy_synthesize(self, text: str, speed: float = 1.0) -> Tuple[str, List[Dict[str, Union[str, float]]]]:
        """
        Create a dummy audio file and word timings when TTS is not available.
        
        Args:
            text: The text to synthesize.
            speed: The speed factor.
            
        Returns:
            Tuple of (audio_path, word_timings)
        """
        # Create a simple sine wave as dummy audio
        sample_rate = 24000  # Kokoro's sample rate
        duration = len(text.split()) * 0.3 / speed  # Rough estimate: 0.3 seconds per word
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Save to a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', dir=self.temp_dir, delete=False)
        temp_file.close()
        sf.write(temp_file.name, audio, sample_rate)
        
        # Create word timings
        words = text.split()
        word_timings = []
        word_duration = duration / len(words) if words else 0
        current_time = 0.0
        
        for word in words:
            word_timings.append({
                "word": word,
                "start": current_time,
                "end": current_time + word_duration
            })
            current_time += word_duration
        
        self.word_timings = word_timings
        
        return temp_file.name, word_timings
    
    def _extract_word_timings(self, words: List[str], duration: float) -> List[Dict[str, Union[str, float]]]:
        """
        Extract word timings based on words and duration.
        
        Args:
            words: List of words.
            duration: The duration of the audio in seconds.
            
        Returns:
            List of dictionaries with word timing information.
        """
        # Calculate average word duration
        word_duration = duration / len(words) if words else 0
        
        timings = []
        current_time = 0.0
        
        for word in words:
            # Adjust duration based on word length (simple heuristic)
            # Longer words get more time, with some randomness for natural variation
            word_length_factor = 0.5 + 0.5 * len(word) / 5
            random_factor = 0.9 + 0.2 * np.random.random()  # Between 0.9 and 1.1
            adjusted_duration = word_duration * word_length_factor * random_factor
            
            timings.append({
                "word": word,
                "start": current_time,
                "end": current_time + adjusted_duration
            })
            
            current_time += adjusted_duration
        
        # Normalize to ensure the last word ends at the total duration
        if timings:
            scale_factor = duration / timings[-1]["end"]
            for timing in timings:
                timing["start"] *= scale_factor
                timing["end"] *= scale_factor
        
        return timings
    
    def get_word_at_position(self, position: float) -> Dict[str, Union[str, float]]:
        """
        Get the word at the given position in the audio.
        
        Args:
            position: Position in seconds.
            
        Returns:
            The word timing dictionary or None if not found.
        """
        for timing in self.word_timings:
            if timing["start"] <= position <= timing["end"]:
                return timing
        return None
    
    def play_audio(self, text: str, voice: str = "af_sarah", speed: float = 1.0, stream: bool = True):
        """
        Play audio directly using sounddevice.
        
        Args:
            text: The text to synthesize.
            voice: The voice to use.
            speed: The speed factor.
            stream: Whether to stream the audio or play it all at once.
        """
        self.stop_requested = False
        self.pause_requested = False
        
        if self.audio_thread and self.audio_thread.is_alive():
            return
            
        self.audio_thread = threading.Thread(
            target=self._process_audio,
            args=(text, voice, speed, stream)
        )
        self.audio_thread.start()
    
    def _process_audio(self, text: str, voice: str, speed: float, stream: bool):
        """
        Process and play audio in a separate thread.
        
        Args:
            text: The text to synthesize.
            voice: The voice to use.
            speed: The speed factor.
            stream: Whether to stream the audio or play it all at once.
        """
        if not self.load_model():
            return
            
        try:
            if stream:
                asyncio.run(self._play_stream(text, voice, speed))
            else:
                samples, sample_rate = self.kokoro.create(
                    text, voice=voice, speed=speed, lang="en-us"
                )
                sd.play(samples, sample_rate)
                sd.wait()
        except Exception as e:
            warnings.warn(f"Failed to play audio: {str(e)}")
    
    async def _play_stream(self, text: str, voice: str, speed: float):
        """
        Play audio as a stream.
        
        Args:
            text: The text to synthesize.
            voice: The voice to use.
            speed: The speed factor.
        """
        stream = self.kokoro.create_stream(
            text, voice=voice, speed=speed, lang="en-us"
        )
        async for samples, sr in stream:
            while self.pause_requested:
                await asyncio.sleep(0.1)
            if self.stop_requested:
                return
            sd.play(samples, sr)
            sd.wait()
    
    def stop_audio(self):
        """Stop audio playback."""
        self.stop_requested = True
        sd.stop()
    
    def toggle_pause(self):
        """Toggle pause state."""
        self.pause_requested = not self.pause_requested
    
    def unload_model(self):
        """Unload the model to free memory."""
        self.kokoro = None
