"""
Text-to-Speech engine module for the Audiobook Reader application.
Uses the Kokoro ONNX model for speech synthesis with timing information.
"""

import os
import tempfile
import warnings
import threading
import asyncio
import queue
import time
from typing import Dict, List, Tuple, Union, Optional, Generator, Any

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
        self.chunks_dir = os.path.join(self.temp_dir, 'chunks')

        # Ensure directories exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.chunks_dir, exist_ok=True)

        self.kokoro = None
        self.current_text = ""
        self.word_timings = []
        self.stop_requested = False
        self.pause_requested = False
        self.audio_thread = None

        # For progressive playback
        self.chunk_size = 100  # Smaller chunks for faster initial playback
        self.audio_queue = queue.Queue()
        self.synthesis_complete = False
        self.current_position = 0.0
        self.chunk_files = []  # Keep track of chunk files

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

    def _generate_dummy_audio(self, text: str, speed: float = 1.0) -> Tuple[np.ndarray, List[Dict[str, Union[str, float]]], float]:
        """
        Generate dummy audio and word timings when TTS is not available.

        Args:
            text: The text to synthesize.
            speed: The speed factor.

        Returns:
            Tuple of (audio_samples, word_timings, duration)
        """
        # Create a simple sine wave as dummy audio
        duration = len(text.split()) * 0.3 / speed  # Rough estimate: 0.3 seconds per word
        t = np.linspace(0, duration, int(self.SAMPLE_RATE * duration), False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

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

        return audio, word_timings, duration

    def _dummy_synthesize(self, text: str, speed: float = 1.0) -> Tuple[str, List[Dict[str, Union[str, float]]]]:
        """
        Create a dummy audio file and word timings when TTS is not available.

        Args:
            text: The text to synthesize.
            speed: The speed factor.

        Returns:
            Tuple of (audio_path, word_timings)
        """
        # Generate dummy audio
        audio, word_timings, _ = self._generate_dummy_audio(text, speed)

        # Save to a file in the chunks directory
        chunk_filename = f"dummy_{int(time.time())}.wav"
        chunk_path = os.path.join(self.chunks_dir, chunk_filename)
        sf.write(chunk_path, audio, self.SAMPLE_RATE)

        self.word_timings = word_timings
        self.chunk_files.append(chunk_path)

        return chunk_path, word_timings

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

    def synthesize_and_play_progressively(self, text: str, voice: str = "af_sarah", speed: float = 1.0, callback=None):
        """
        Synthesize speech from text and start playing as soon as the first chunk is ready.

        Args:
            text: The text to synthesize.
            voice: The voice to use.
            speed: The speed factor (1.0 is normal speed).
            callback: Optional callback function to call when a chunk is ready.
                     The callback receives (chunk_index, total_chunks, audio_path, word_timings).
        """
        # Store the current text
        self.current_text = text
        self.stop_requested = False
        self.pause_requested = False
        self.synthesis_complete = False
        self.current_position = 0.0

        # Clear the queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        # Clean up old chunk files
        self._clean_chunk_files()

        # Start the synthesis thread
        synthesis_thread = threading.Thread(
            target=self._synthesize_chunks,
            args=(text, voice, speed, callback)
        )
        synthesis_thread.daemon = True
        synthesis_thread.start()

        # Start the playback thread
        playback_thread = threading.Thread(
            target=self._play_chunks
        )
        playback_thread.daemon = True
        playback_thread.start()

        return synthesis_thread, playback_thread

    def _clean_chunk_files(self):
        """Clean up old chunk files."""
        # Clear the chunk files list
        self.chunk_files = []

        # Remove old chunk files
        for file in os.listdir(self.chunks_dir):
            if file.endswith('.wav'):
                try:
                    os.remove(os.path.join(self.chunks_dir, file))
                except Exception as e:
                    warnings.warn(f"Failed to remove chunk file {file}: {str(e)}")

    def _synthesize_chunks(self, text: str, voice: str, speed: float, callback=None):
        """
        Synthesize text in chunks and add them to the queue.

        Args:
            text: The text to synthesize.
            voice: The voice to use.
            speed: The speed factor.
            callback: Optional callback function.
        """
        if not self.load_model():
            # Use fallback synthesis
            audio_path, word_timings = self._dummy_synthesize(text, speed)
            self.audio_queue.put((0, 1, audio_path, word_timings))
            if callback:
                callback(0, 1, audio_path, word_timings)
            self.synthesis_complete = True
            return

        # Split text into chunks
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size):
            chunk = ' '.join(words[i:i+self.chunk_size])
            chunks.append(chunk)

        total_chunks = len(chunks)
        all_word_timings = []
        time_offset = 0.0

        for i, chunk in enumerate(chunks):
            if self.stop_requested:
                break

            try:
                # Generate speech for this chunk
                print(f"Synthesizing chunk {i+1}/{total_chunks}")

                # Create a file for the audio in the chunks directory
                chunk_filename = f"chunk_{i:04d}_{int(time.time())}.wav"
                chunk_path = os.path.join(self.chunks_dir, chunk_filename)
                self.chunk_files.append(chunk_path)

                # Generate speech
                samples, sample_rate = self.kokoro.create(
                    chunk, voice=voice, speed=speed, lang="en-us"
                )

                # Save the audio to a file
                sf.write(chunk_path, samples, sample_rate)

                # Calculate duration
                duration = len(samples) / sample_rate

                # Extract word timings
                chunk_words = chunk.split()
                word_timings = self._extract_word_timings(chunk_words, duration)

                # Adjust timings based on offset
                for timing in word_timings:
                    timing["start"] += time_offset
                    timing["end"] += time_offset

                # Add to the queue
                self.audio_queue.put((i, total_chunks, chunk_path, word_timings))

                # Update the global word timings
                all_word_timings.extend(word_timings)

                # Update the time offset
                time_offset += duration

                # Call the callback if provided
                if callback:
                    callback(i, total_chunks, chunk_path, word_timings)

            except Exception as e:
                warnings.warn(f"Failed to synthesize chunk {i}: {str(e)}")
                # Use fallback for this chunk
                # Create a file for the audio in the chunks directory
                chunk_filename = f"chunk_dummy_{i:04d}_{int(time.time())}.wav"
                chunk_path = os.path.join(self.chunks_dir, chunk_filename)
                self.chunk_files.append(chunk_path)

                # Generate dummy audio
                dummy_audio, dummy_timings, dummy_duration = self._generate_dummy_audio(chunk, speed)

                # Save to file
                sf.write(chunk_path, dummy_audio, self.SAMPLE_RATE)

                # Adjust timings based on offset
                for timing in dummy_timings:
                    timing["start"] += time_offset
                    timing["end"] += time_offset

                # Add to the queue
                self.audio_queue.put((i, total_chunks, chunk_path, dummy_timings))

                # Update the global word timings
                all_word_timings.extend(dummy_timings)

                # Update the time offset
                time_offset += dummy_duration

                # Call the callback if provided
                if callback:
                    callback(i, total_chunks, chunk_path, dummy_timings)

        # Store the complete word timings
        self.word_timings = all_word_timings

        # Mark synthesis as complete
        self.synthesis_complete = True

    def _play_chunks(self):
        """Play audio chunks from the queue."""
        current_chunk = 0
        chunk_start_time = 0.0

        while not (self.synthesis_complete and self.audio_queue.empty()) and not self.stop_requested:
            try:
                # Get the next chunk from the queue
                chunk_index, total_chunks, audio_path, word_timings = self.audio_queue.get(timeout=0.5)

                # Skip chunks that are out of order
                if chunk_index < current_chunk:
                    continue

                # Wait for the correct chunk
                while chunk_index > current_chunk and not self.stop_requested:
                    # Put the chunk back in the queue
                    self.audio_queue.put((chunk_index, total_chunks, audio_path, word_timings))
                    # Wait a bit
                    time.sleep(0.1)
                    # Try to get the correct chunk
                    chunk_index, total_chunks, audio_path, word_timings = self.audio_queue.get(timeout=0.5)

                # Play the audio
                if not self.stop_requested:
                    print(f"Playing chunk {chunk_index+1}/{total_chunks}")

                    # Load the audio
                    audio, sr = sf.read(audio_path)

                    # Calculate chunk duration
                    chunk_duration = len(audio) / sr

                    # Set the start time for this chunk
                    chunk_start_time = word_timings[0]["start"] if word_timings else 0.0

                    # Play the audio
                    sd.play(audio, sr)

                    # Start time for tracking position
                    start_time = time.time()

                    # Wait for the audio to finish
                    while sd.get_stream().active and not self.stop_requested:
                        # Update current position
                        if not self.pause_requested:
                            elapsed = time.time() - start_time
                            self.current_position = chunk_start_time + elapsed

                        # Handle pause
                        if self.pause_requested:
                            # Save position before pausing
                            pause_time = time.time()
                            elapsed_before_pause = pause_time - start_time

                            # Stop playback
                            sd.stop()

                            # Wait while paused
                            while self.pause_requested and not self.stop_requested:
                                time.sleep(0.1)

                            # Resume if not stopped
                            if not self.stop_requested:
                                # Calculate remaining audio
                                remaining_time = chunk_duration - elapsed_before_pause
                                if remaining_time > 0:
                                    # Calculate position to resume from
                                    resume_pos = int(elapsed_before_pause * sr)
                                    if resume_pos < len(audio):
                                        # Play remaining audio
                                        sd.play(audio[resume_pos:], sr)
                                        # Update start time for correct position tracking
                                        start_time = time.time() - elapsed_before_pause

                        time.sleep(0.1)

                    # Update the current chunk
                    current_chunk = chunk_index + 1

            except queue.Empty:
                # No chunks available yet, wait a bit
                time.sleep(0.1)
            except Exception as e:
                warnings.warn(f"Error playing chunk: {str(e)}")
                time.sleep(0.1)

    def unload_model(self):
        """Unload the model to free memory."""
        self.kokoro = None
