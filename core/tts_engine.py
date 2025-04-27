"""
Text-to-Speech engine module for the Audiobook Reader application.
Uses the Kokoro model for speech synthesis with timing information.
Implements streaming and windowed text processing for efficient handling of large texts.
"""

import os
import tempfile
import re
import time
import warnings
from typing import Dict, List, Optional, Tuple, Union, Generator, Any
from collections import deque

import numpy as np
import soundfile as sf

# Try to import torch and transformers, but handle import errors gracefully
try:
    import torch
    from transformers import AutoProcessor, AutoModel
    TRANSFORMERS_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    warnings.warn(f"Failed to import torch or transformers: {str(e)}. TTS functionality will be limited.")
    TRANSFORMERS_AVAILABLE = False


class TextChunk:
    """Represents a chunk of text with its associated audio and timing data."""

    def __init__(self, text: str, chunk_id: int):
        """
        Initialize a text chunk.

        Args:
            text: The text content of the chunk.
            chunk_id: Unique identifier for the chunk.
        """
        self.text = text
        self.chunk_id = chunk_id
        self.audio_path = None
        self.word_timings = []
        self.processed = False
        self.start_time = 0.0  # Start time relative to the entire audio
        self.duration = 0.0

    def set_audio_data(self, audio_path: str, word_timings: List[Dict[str, Union[str, float]]], duration: float):
        """
        Set the audio data for this chunk.

        Args:
            audio_path: Path to the audio file.
            word_timings: List of word timing dictionaries.
            duration: Duration of the audio in seconds.
        """
        self.audio_path = audio_path
        self.word_timings = word_timings
        self.duration = duration
        self.processed = True

    def adjust_timings(self, start_time: float):
        """
        Adjust word timings based on the chunk's start time in the full audio.

        Args:
            start_time: The start time of this chunk in the full audio.
        """
        self.start_time = start_time

        # Adjust all word timings
        for timing in self.word_timings:
            timing["start"] += start_time
            timing["end"] += start_time


class TTSEngine:
    """Interface for the Text-to-Speech model (Kokoro) with streaming support."""

    MODEL_ID = "hexgrad/Kokoro-82M"
    CHUNK_SIZE = 100  # Number of words per chunk
    MAX_CACHE_CHUNKS = 10  # Maximum number of chunks to keep in cache

    def __init__(self, model_dir: Optional[str] = None, temp_dir: Optional[str] = None):
        """
        Initialize the TTS engine.

        Args:
            model_dir: Directory to store the model. If None, uses the default.
            temp_dir: Directory for temporary files. If None, uses the default.
        """
        self.model_dir = model_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'kokoro')
        self.temp_dir = temp_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'temp')

        # Ensure directories exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Cache for processed chunks
        self.chunk_cache = deque(maxlen=self.MAX_CACHE_CHUNKS)
        self.current_text = ""
        self.chunks = []
        self.combined_audio_path = None

    def load_model(self):
        """Load the Kokoro model."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Cannot load model: transformers or torch is not available. "
                             "Please install the required dependencies.")

        if self.processor is None or self.model is None:
            # Load the model and processor
            self.processor = AutoProcessor.from_pretrained(self.MODEL_ID, cache_dir=self.model_dir)
            self.model = AutoModel.from_pretrained(self.MODEL_ID, cache_dir=self.model_dir).to(self.device)

    def split_text_into_chunks(self, text: str) -> List[TextChunk]:
        """
        Split text into chunks of approximately CHUNK_SIZE words.

        Args:
            text: The text to split.

        Returns:
            List of TextChunk objects.
        """
        # Split text into words while preserving punctuation and spacing
        words = re.findall(r'\S+|\s+', text)

        chunks = []
        current_chunk = []
        word_count = 0
        chunk_id = 0

        for word in words:
            current_chunk.append(word)

            # Only count non-whitespace as words
            if not word.isspace():
                word_count += 1

            # When we reach CHUNK_SIZE words, create a new chunk
            if word_count >= self.CHUNK_SIZE:
                chunk_text = ''.join(current_chunk)
                chunks.append(TextChunk(chunk_text, chunk_id))
                chunk_id += 1
                current_chunk = []
                word_count = 0

        # Add the last chunk if there's any text left
        if current_chunk:
            chunk_text = ''.join(current_chunk)
            chunks.append(TextChunk(chunk_text, chunk_id))

        return chunks

    def create_stream(self, text: str, voice: str = "default", speed: float = 1.0) -> Generator[Dict[str, Any], None, None]:
        """
        Create a stream of audio chunks from text.

        Args:
            text: The text to synthesize.
            voice: The voice to use.
            speed: The speed factor (1.0 is normal speed).

        Yields:
            Dictionary with chunk information:
                - chunk_id: The chunk ID
                - audio_path: Path to the audio file
                - word_timings: List of word timing dictionaries
                - duration: Duration of the chunk in seconds
        """
        # Check if transformers is available
        if not TRANSFORMERS_AVAILABLE:
            # Use dummy synthesis for the entire text
            audio_path, word_timings = self._dummy_synthesize(text, speed)
            yield {
                "chunk_id": 0,
                "audio_path": audio_path,
                "word_timings": word_timings,
                "duration": len(word_timings) * 0.3 / speed if word_timings else 0
            }
            return

        if self.processor is None or self.model is None:
            self.load_model()

        # Split text into chunks
        chunks = self.split_text_into_chunks(text)

        # Process each chunk
        for chunk in chunks:
            # Check if the chunk is already in cache
            cached_chunk = self._get_cached_chunk(chunk.text)
            if cached_chunk:
                yield {
                    "chunk_id": chunk.chunk_id,
                    "audio_path": cached_chunk.audio_path,
                    "word_timings": cached_chunk.word_timings,
                    "duration": cached_chunk.duration
                }
                continue

            # Process the chunk
            audio_path, word_timings, duration = self._synthesize_chunk(chunk.text, voice, speed)

            # Update the chunk
            chunk.set_audio_data(audio_path, word_timings, duration)

            # Add to cache
            self._add_to_cache(chunk)

            # Yield the chunk information
            yield {
                "chunk_id": chunk.chunk_id,
                "audio_path": audio_path,
                "word_timings": word_timings,
                "duration": duration
            }

    def synthesize(self, text: str, voice: str = "default", speed: float = 1.0) -> Tuple[str, List[Dict[str, Union[str, float]]]]:
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

        # Check if transformers is available
        if not TRANSFORMERS_AVAILABLE:
            return self._dummy_synthesize(text, speed)

        # Split text into chunks
        self.chunks = self.split_text_into_chunks(text)

        # Process all chunks
        all_audio_paths = []
        all_word_timings = []
        current_time = 0.0

        for chunk in self.chunks:
            # Check if the chunk is already in cache
            cached_chunk = self._get_cached_chunk(chunk.text)
            if cached_chunk:
                audio_path = cached_chunk.audio_path
                word_timings = cached_chunk.word_timings
                duration = cached_chunk.duration
            else:
                # Process the chunk
                audio_path, word_timings, duration = self._synthesize_chunk(chunk.text, voice, speed)

                # Update the chunk
                chunk.set_audio_data(audio_path, word_timings, duration)

                # Add to cache
                self._add_to_cache(chunk)

            # Adjust timings based on position in the full audio
            chunk.adjust_timings(current_time)
            current_time += chunk.duration

            # Collect data
            all_audio_paths.append(audio_path)
            all_word_timings.extend(chunk.word_timings)

        # Combine all audio files into one
        combined_audio_path = self._combine_audio_files(all_audio_paths)
        self.combined_audio_path = combined_audio_path

        return combined_audio_path, all_word_timings

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
        sample_rate = 22050
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

        # Store as a single chunk for consistency
        self.chunks = [TextChunk(text, 0)]
        self.chunks[0].set_audio_data(temp_file.name, word_timings, duration)

        return temp_file.name, word_timings

    def _synthesize_chunk(self, text: str, voice: str, speed: float) -> Tuple[str, List[Dict[str, Union[str, float]]], float]:
        """
        Synthesize speech for a chunk of text.

        Args:
            text: The text to synthesize.
            voice: The voice to use.
            speed: The speed factor.

        Returns:
            Tuple of (audio_path, word_timings, duration)
        """
        if not TRANSFORMERS_AVAILABLE:
            # Create a dummy audio for this chunk
            return self._dummy_synthesize_chunk(text, speed)

        # Process the text
        inputs = self.processor(
            text=text,
            voice_preset=voice,
            return_tensors="pt"
        ).to(self.device)

        # Generate speech
        with torch.no_grad():
            output = self.model.generate(**inputs, speaker_id=0)

        # Get the audio data
        audio = output.cpu().numpy().squeeze()
        sample_rate = self.model.config.sampling_rate

        # Adjust speed if needed
        if speed != 1.0:
            # This is a simple approach - for better quality, consider using librosa
            indices = torch.arange(0, len(audio), speed)
            indices = indices.to(torch.long)
            audio = audio[indices.to('cpu').numpy()]

        # Save to a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', dir=self.temp_dir, delete=False)
        temp_file.close()
        sf.write(temp_file.name, audio, sample_rate)

        # Calculate duration
        duration = len(audio) / sample_rate

        # Extract word timings
        word_timings = self._extract_word_timings(text, duration)

        return temp_file.name, word_timings, duration

    def _dummy_synthesize_chunk(self, text: str, speed: float) -> Tuple[str, List[Dict[str, Union[str, float]]], float]:
        """
        Create a dummy audio file for a chunk when TTS is not available.

        Args:
            text: The text to synthesize.
            speed: The speed factor.

        Returns:
            Tuple of (audio_path, word_timings, duration)
        """
        # Create a simple sine wave as dummy audio
        sample_rate = 22050
        duration = len(text.split()) * 0.3 / speed  # Rough estimate: 0.3 seconds per word
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

        # Save to a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', dir=self.temp_dir, delete=False)
        temp_file.close()
        sf.write(temp_file.name, audio, sample_rate)

        # Extract word timings
        word_timings = self._extract_word_timings(text, duration)

        return temp_file.name, word_timings, duration

    def _extract_word_timings(self, text: str, duration: float) -> List[Dict[str, Union[str, float]]]:
        """
        Extract word timings based on text and duration.

        Args:
            text: The input text.
            duration: The duration of the audio in seconds.

        Returns:
            List of dictionaries with word timing information.
        """
        # Split text into words while preserving punctuation
        words = re.findall(r'\S+', text)

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

    def _combine_audio_files(self, audio_paths: List[str]) -> str:
        """
        Combine multiple audio files into one.

        Args:
            audio_paths: List of paths to audio files.

        Returns:
            Path to the combined audio file.
        """
        if not audio_paths:
            return None

        if len(audio_paths) == 1:
            return audio_paths[0]

        # Create a temporary file for the combined audio
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', dir=self.temp_dir, delete=False)
        temp_file.close()

        # Load and combine audio data
        combined_audio = None
        sample_rate = None

        for path in audio_paths:
            audio, sr = sf.read(path)

            if combined_audio is None:
                combined_audio = audio
                sample_rate = sr
            else:
                combined_audio = np.concatenate((combined_audio, audio))

        # Save the combined audio
        sf.write(temp_file.name, combined_audio, sample_rate)

        return temp_file.name

    def _get_cached_chunk(self, text: str) -> Optional[TextChunk]:
        """
        Get a chunk from the cache if it exists.

        Args:
            text: The text to look for.

        Returns:
            The cached chunk or None if not found.
        """
        for chunk in self.chunk_cache:
            if chunk.text == text:
                return chunk
        return None

    def _add_to_cache(self, chunk: TextChunk):
        """
        Add a chunk to the cache.

        Args:
            chunk: The chunk to add.
        """
        # Check if the chunk is already in cache
        for i, cached_chunk in enumerate(self.chunk_cache):
            if cached_chunk.text == chunk.text:
                # Replace the existing chunk
                self.chunk_cache[i] = chunk
                return

        # Add to cache
        self.chunk_cache.append(chunk)

    def update_chunk(self, chunk_id: int, new_text: str, voice: str = "default", speed: float = 1.0) -> Tuple[str, List[Dict[str, Union[str, float]]]]:
        """
        Update a specific chunk with new text and regenerate audio.

        Args:
            chunk_id: The ID of the chunk to update.
            new_text: The new text for the chunk.
            voice: The voice to use.
            speed: The speed factor.

        Returns:
            Tuple of (combined_audio_path, all_word_timings)
        """
        # Find the chunk
        for i, chunk in enumerate(self.chunks):
            if chunk.chunk_id == chunk_id:
                # Create a new chunk with the updated text
                new_chunk = TextChunk(new_text, chunk_id)

                # Process the chunk
                audio_path, word_timings, duration = self._synthesize_chunk(new_text, voice, speed)

                # Update the chunk
                new_chunk.set_audio_data(audio_path, word_timings, duration)

                # Replace the old chunk
                self.chunks[i] = new_chunk

                # Add to cache
                self._add_to_cache(new_chunk)

                # Recalculate timings for all chunks
                self._recalculate_timings()

                # Regenerate combined audio
                all_audio_paths = [chunk.audio_path for chunk in self.chunks]
                all_word_timings = []
                for chunk in self.chunks:
                    all_word_timings.extend(chunk.word_timings)

                combined_audio_path = self._combine_audio_files(all_audio_paths)
                self.combined_audio_path = combined_audio_path

                return combined_audio_path, all_word_timings

        # Chunk not found
        raise ValueError(f"Chunk with ID {chunk_id} not found")

    def _recalculate_timings(self):
        """Recalculate timings for all chunks based on their position in the sequence."""
        current_time = 0.0

        for chunk in self.chunks:
            chunk.adjust_timings(current_time)
            current_time += chunk.duration

    def get_chunk_for_position(self, position: float) -> Optional[TextChunk]:
        """
        Get the chunk that contains the given position in the audio.

        Args:
            position: Position in seconds.

        Returns:
            The chunk or None if not found.
        """
        for chunk in self.chunks:
            if chunk.start_time <= position < (chunk.start_time + chunk.duration):
                return chunk
        return None

    def get_word_at_position(self, position: float) -> Optional[Dict[str, Union[str, float]]]:
        """
        Get the word at the given position in the audio.

        Args:
            position: Position in seconds.

        Returns:
            The word timing dictionary or None if not found.
        """
        for timing in self._get_all_word_timings():
            if timing["start"] <= position <= timing["end"]:
                return timing
        return None

    def _get_all_word_timings(self) -> List[Dict[str, Union[str, float]]]:
        """
        Get all word timings from all chunks.

        Returns:
            List of all word timing dictionaries.
        """
        all_timings = []
        for chunk in self.chunks:
            all_timings.extend(chunk.word_timings)
        return all_timings

    def unload_model(self):
        """Unload the model to free memory."""
        self.processor = None
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
