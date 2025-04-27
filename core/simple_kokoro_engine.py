"""
Simple Kokoro TTS engine module for the Audiobook Reader application.
This is a simplified version that uses the Kokoro pipeline directly.
"""

import os
import tempfile
import warnings
from typing import Dict, List, Tuple, Union

import numpy as np
import soundfile as sf

# Try to import kokoro, but handle import errors gracefully
try:
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Failed to import kokoro: {str(e)}. TTS functionality will be limited.")
    KOKORO_AVAILABLE = False


class SimpleKokoroEngine:
    """A simple interface for the Kokoro TTS pipeline."""
    
    SAMPLE_RATE = 24000  # Kokoro's sample rate
    
    def __init__(self, temp_dir: str = None):
        """
        Initialize the TTS engine.
        
        Args:
            temp_dir: Directory for temporary files. If None, uses the default.
        """
        self.temp_dir = temp_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'temp')
        
        # Ensure directories exist
        os.makedirs(self.temp_dir, exist_ok=True)
        
        self.pipeline = None
        self.current_text = ""
        self.word_timings = []
    
    def load_model(self):
        """Load the Kokoro model."""
        if not KOKORO_AVAILABLE:
            warnings.warn("Kokoro is not available. Using fallback synthesis.")
            return False
            
        if self.pipeline is None:
            try:
                # Initialize the Kokoro pipeline
                self.pipeline = KPipeline(lang_code='a')  # 'a' for auto-detect
                print("Kokoro pipeline initialized successfully")
                return True
            except Exception as e:
                warnings.warn(f"Failed to initialize Kokoro pipeline: {str(e)}. Using fallback synthesis.")
                self.pipeline = None
                return False
        
        return True
    
    def synthesize(self, text: str, voice: str = "af_heart", speed: float = 1.0) -> Tuple[str, List[Dict[str, Union[str, float]]]]:
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
            # Generate speech using Kokoro pipeline
            print(f"Synthesizing speech with voice: {voice}, speed: {speed}")
            
            # Create a temporary file for the audio
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', dir=self.temp_dir, delete=False)
            temp_file.close()
            
            # Process the text with Kokoro
            all_audio = []
            word_positions = []
            word_count = 0
            
            # Use the pipeline to generate audio
            generator = self.pipeline(text, voice=voice)
            
            for i, (gs, ps, audio) in enumerate(generator):
                # Collect audio segments
                all_audio.append(audio)
                
                # Extract words from this segment
                segment_text = text.split()[word_count:word_count + len(gs)]
                word_count += len(gs)
                
                # Store word positions
                for word in segment_text:
                    word_positions.append(word)
            
            # Combine all audio segments
            if all_audio:
                combined_audio = np.concatenate(all_audio)
                
                # Apply speed adjustment if needed
                if speed != 1.0:
                    # Simple resampling for speed adjustment
                    indices = np.round(np.linspace(0, len(combined_audio) - 1, int(len(combined_audio) / speed))).astype(int)
                    combined_audio = combined_audio[indices]
                
                # Save the combined audio
                sf.write(temp_file.name, combined_audio, self.SAMPLE_RATE)
                
                # Calculate duration
                duration = len(combined_audio) / self.SAMPLE_RATE
                
                # Extract word timings
                word_timings = self._extract_word_timings(word_positions, duration)
                self.word_timings = word_timings
                
                return temp_file.name, word_timings
            else:
                # Fallback if no audio was generated
                return self._dummy_synthesize(text, speed)
                
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
        sample_rate = self.SAMPLE_RATE
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
    
    def unload_model(self):
        """Unload the model to free memory."""
        self.pipeline = None
