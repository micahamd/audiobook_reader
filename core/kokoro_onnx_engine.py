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

    # Constants
    SAMPLE_RATE = 24000  # Kokoro's sample rate

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
            # Try to use create_stream to get direct timing information
            try:
                print(f"Synthesizing speech with voice: {voice}, speed: {speed} using streaming with direct timings")

                # Create a temporary file for the audio
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', dir=self.temp_dir, delete=False)
                temp_file.close()

                # Create temporary lists to collect all samples and timings
                all_samples = []
                all_timings = []

                # Create an event loop for async operations
                loop = None
                try:
                    # Try to get the current event loop
                    try:
                        loop = asyncio.get_running_loop()
                        print("Using existing event loop")
                    except RuntimeError:
                        # No running event loop, create a new one
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        print("Created new event loop")

                    # Define the async function to process the stream
                    sample_rate_container = [self.SAMPLE_RATE]  # Use a container to store the sample rate

                    async def process_stream():
                        try:
                            stream = self.kokoro.create_stream(
                                text, voice=voice, speed=speed, lang="en-us"
                            )

                            async for result in stream:
                                # Check if we're being shut down
                                if self.stop_requested:
                                    print("Stop requested during stream processing")
                                    break

                                # Handle both 2-value and 3-value tuples
                                if len(result) == 3:
                                    samples, sr, timings = result
                                elif len(result) == 2:
                                    samples, sr = result
                                    timings = []
                                else:
                                    print(f"Unexpected result format: {result}")
                                    continue

                                all_samples.append(samples)
                                # Store the sample rate for later use
                                sample_rate_container[0] = sr
                                if timings:
                                    all_timings.extend(timings)
                        except asyncio.CancelledError:
                            print("Stream processing was cancelled")
                            raise
                        except Exception as e:
                            print(f"Error in process_stream: {str(e)}")
                            raise

                    # Run the async function
                    if loop.is_running():
                        # If the loop is already running, create a task
                        print("Loop is already running, creating task")
                        task = asyncio.create_task(process_stream())
                        # Wait for the task to complete
                        while not task.done():
                            time.sleep(0.1)
                            if self.stop_requested:
                                print("Stop requested, cancelling task")
                                task.cancel()
                                break
                    else:
                        # Otherwise, run until complete
                        print("Running process_stream in event loop")
                        loop.run_until_complete(process_stream())
                except Exception as e:
                    print(f"Error in event loop handling: {str(e)}")
                    raise
                finally:
                    # Only close the loop if we created it
                    if loop and not loop.is_running():
                        try:
                            loop.close()
                            print("Closed event loop")
                        except Exception as e:
                            print(f"Error closing event loop: {str(e)}")

                # Combine all samples
                if all_samples:
                    samples = np.concatenate(all_samples)
                    sample_rate = sample_rate_container[0]

                    # Save the audio to a file
                    sf.write(temp_file.name, samples, sample_rate)

                    # Process the direct timing information
                    if all_timings:
                        print(f"Using direct timing information: {len(all_timings)} words")
                        word_timings_list = self._process_direct_timings(all_timings, text)

                        # Store both the list and dictionary versions
                        self.word_timings_list = word_timings_list
                        self.word_timings = self._convert_word_timings_to_dict(word_timings_list)

                        print(f"Converted word timings to dictionary with {len(self.word_timings)} entries")
                        return temp_file.name, word_timings_list
                    else:
                        # If no timings, fall back to heuristic method
                        raise Exception("No timing information received from stream")
                else:
                    # If no samples, fall back to create method
                    raise Exception("No audio samples collected from stream")

            except Exception as stream_error:
                print(f"Error using create_stream: {str(stream_error)}. Falling back to create method.")

                # Generate speech using Kokoro's create method
                print(f"Synthesizing speech with voice: {voice}, speed: {speed} using create method")

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

                # Extract word timings using heuristic method
                word_timings_list = self._extract_word_timings(text.split(), duration)

                # Store both the list and dictionary versions
                self.word_timings_list = word_timings_list
                self.word_timings = self._convert_word_timings_to_dict(word_timings_list)

                print(f"Using heuristic word timings, converted to dictionary with {len(self.word_timings)} entries")
                return temp_file.name, word_timings_list

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

        # Store both the list and dictionary versions
        self.word_timings_list = word_timings
        self.word_timings = self._convert_word_timings_to_dict(word_timings)

        self.chunk_files.append(chunk_path)

        return chunk_path, word_timings

    def _extract_word_timings(self, words: List[str], duration: float) -> List[Dict[str, Union[str, float]]]:
        """
        Extract word timings based on words and duration.
        This is a fallback method when direct timing information is not available.

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

        # Track position in the original text
        position = 0

        for word in words:
            # Adjust duration based on word length (simple heuristic)
            # Longer words get more time, with some randomness for natural variation
            word_length_factor = 0.5 + 0.5 * len(word) / 5
            random_factor = 0.9 + 0.2 * np.random.random()  # Between 0.9 and 1.1
            adjusted_duration = word_duration * word_length_factor * random_factor

            timings.append({
                "word": word,
                "start": current_time,
                "end": current_time + adjusted_duration,
                "position": position  # Add position information
            })

            current_time += adjusted_duration
            position += len(word) + 1  # +1 for the space

        # Normalize to ensure the last word ends at the total duration
        if timings:
            scale_factor = duration / timings[-1]["end"]
            for timing in timings:
                timing["start"] *= scale_factor
                timing["end"] *= scale_factor

        return timings

    def _process_direct_timings(self, kokoro_timings: List[Dict], text: str = None) -> List[Dict[str, Union[str, float]]]:
        """
        Process direct timing information from Kokoro's create_stream method.

        Args:
            kokoro_timings: List of timing dictionaries from Kokoro.
            text: The original text for position tracking.

        Returns:
            List of dictionaries with word timing information in our format.
        """
        word_timings = []

        # If we have the original text, try to find word positions
        word_positions = {}
        if text:
            # Simple approach to find word positions in the text
            current_pos = 0
            for word in text.split():
                # Find the word in the text starting from current_pos
                word_pos = text.find(word, current_pos)
                if word_pos >= 0:
                    word_positions[word] = word_pos
                    current_pos = word_pos + len(word)

        for timing in kokoro_timings:
            # Kokoro timings have 'word', 'start', and 'end' keys
            word = timing.get("word", "")
            start = timing.get("start", 0.0)
            end = timing.get("end", 0.0)

            timing_info = {
                "word": word,
                "start": start,
                "end": end
            }

            # Add position information if available
            if word in word_positions:
                timing_info["position"] = word_positions[word]

            word_timings.append(timing_info)

        return word_timings

    def _convert_word_timings_to_dict(self, word_timings_list: List[Dict[str, Union[str, float]]]) -> Dict[str, float]:
        """
        Convert a list of word timing dictionaries to a position-to-time dictionary.
        This maintains compatibility with code that expects self.word_timings to be a dictionary.

        Args:
            word_timings_list: List of dictionaries with word timing information.

        Returns:
            Dictionary mapping position (as string) to start time (float).
        """
        position_to_time = {}

        for timing in word_timings_list:
            # Only include timings that have position information
            if "position" in timing:
                # Convert position to string to ensure consistent key type
                position = str(timing["position"])
                start_time = float(timing["start"])
                position_to_time[position] = start_time

        # If we don't have any positions, create a simple mapping based on index
        if not position_to_time and word_timings_list:
            print("Warning: No position information in word timings. Creating simple index-based mapping.")
            for i, timing in enumerate(word_timings_list):
                # Estimate position as 6 characters per word
                estimated_position = i * 6
                # Convert to string for consistent key type
                position_to_time[str(estimated_position)] = float(timing["start"])

        # Print some debug information
        if position_to_time:
            print(f"Created word timings dictionary with {len(position_to_time)} entries")
            # Print a few sample entries
            sample_keys = list(position_to_time.keys())[:5] if len(position_to_time) > 5 else list(position_to_time.keys())
            print(f"Sample keys: {sample_keys}")
            print(f"Key types: {[type(k).__name__ for k in sample_keys]}")
        else:
            print("Warning: Empty word timings dictionary created")

        return position_to_time

    def _process_word_timings(self, words, start_time=0.0, text=None):
        """
        Process word timings from the TTS engine.

        Args:
            words: List of words with timing information.
            start_time: Start time offset.
            text: The original text for position tracking.

        Returns:
            List of word timings.
        """
        word_timings = []

        # If we have the original text, try to find word positions
        word_positions = {}
        if text:
            # Simple approach to find word positions in the text
            current_pos = 0
            for word in text.split():
                # Find the word in the text starting from current_pos
                word_pos = text.find(word, current_pos)
                if word_pos >= 0:
                    word_positions[word] = word_pos
                    current_pos = word_pos + len(word)

        for i, word_info in enumerate(words):
            word = word_info["word"]
            start = word_info["start"] + start_time
            end = word_info["end"] + start_time

            timing_info = {
                "word": word,
                "start": start,
                "end": end
            }

            # Add position information if available
            if word in word_positions:
                timing_info["position"] = word_positions[word]
            elif "position" in word_info:
                timing_info["position"] = word_info["position"]

            word_timings.append(timing_info)

        return word_timings

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
        try:
            stream = self.kokoro.create_stream(
                text, voice=voice, speed=speed, lang="en-us"
            )

            # Handle both 2-value and 3-value tuples
            async for result in stream:
                # Handle both 2-value and 3-value tuples
                if len(result) == 3:
                    samples, sr, timings = result
                    # Print timing information for debugging
                    if timings:
                        print(f"Received timing information: {len(timings)} words")
                elif len(result) == 2:
                    samples, sr = result
                    timings = []
                    print("No timing information received (2-value tuple)")
                else:
                    print(f"Unexpected result format: {result}")
                    continue

                while self.pause_requested:
                    await asyncio.sleep(0.1)
                if self.stop_requested:
                    return
                sd.play(samples, sr)
                sd.wait()
        except Exception as e:
            print(f"Error in _play_stream: {str(e)}")
            # Don't re-raise to avoid crashing the application

    def stop_audio(self):
        """Stop audio playback."""
        self.stop_requested = True
        sd.stop()

    def stop(self):
        """Stop playback and reset state."""
        print("Stopping TTS engine playback")
        self.stop_requested = True
        self.pause_requested = False
        self.current_position = 0.0
        sd.stop()

        # Clear any queued audio
        if hasattr(self, 'audio_queue'):
            try:
                while not self.audio_queue.empty():
                    self.audio_queue.get_nowait()
            except Exception as e:
                print(f"Error clearing audio queue: {str(e)}")

    def toggle_pause(self):
        """Toggle pause state."""
        print(f"Toggling pause state from {self.pause_requested} to {not self.pause_requested}")
        self.pause_requested = not self.pause_requested

        # If we're unpausing, make sure we're not also stopped
        if not self.pause_requested:
            self.stop_requested = False

        return self.pause_requested  # Return the new state

    def set_position(self, position: float):
        """
        Set the current playback position.

        Args:
            position: Position in seconds.
        """
        print(f"Setting TTS engine position to {position}s")
        self.current_position = position

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

                # Try to use create_stream to get direct timing information
                try:
                    # Create a temporary list to collect all samples and timings
                    all_samples = []
                    all_timings = []

                    # Create an event loop for async operations
                    loop = None
                    try:
                        # Try to get the current event loop
                        try:
                            loop = asyncio.get_running_loop()
                            print(f"Using existing event loop for chunk {i+1}")
                        except RuntimeError:
                            # No running event loop, create a new one
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            print(f"Created new event loop for chunk {i+1}")

                        # Define the async function to process the stream
                        sample_rate_container = [self.SAMPLE_RATE]  # Use a container to store the sample rate

                        async def process_stream():
                            try:
                                stream = self.kokoro.create_stream(
                                    chunk, voice=voice, speed=speed, lang="en-us"
                                )

                                async for result in stream:
                                    # Check if we're being shut down
                                    if self.stop_requested:
                                        print(f"Stop requested during stream processing for chunk {i+1}")
                                        break

                                    # Handle both 2-value and 3-value tuples
                                    if len(result) == 3:
                                        samples, sr, timings = result
                                    elif len(result) == 2:
                                        samples, sr = result
                                        timings = []
                                    else:
                                        print(f"Unexpected result format: {result}")
                                        continue

                                    all_samples.append(samples)
                                    # Store the sample rate for later use
                                    sample_rate_container[0] = sr
                                    if timings:
                                        all_timings.extend(timings)
                            except asyncio.CancelledError:
                                print(f"Stream processing was cancelled for chunk {i+1}")
                                raise
                            except Exception as e:
                                print(f"Error in process_stream for chunk {i+1}: {str(e)}")
                                raise

                        # Run the async function
                        if loop.is_running():
                            # If the loop is already running, create a task
                            print(f"Loop is already running for chunk {i+1}, creating task")
                            task = asyncio.create_task(process_stream())
                            # Wait for the task to complete
                            while not task.done():
                                time.sleep(0.1)
                                if self.stop_requested:
                                    print(f"Stop requested, cancelling task for chunk {i+1}")
                                    task.cancel()
                                    break
                        else:
                            # Otherwise, run until complete
                            print(f"Running process_stream in event loop for chunk {i+1}")
                            loop.run_until_complete(process_stream())
                    except Exception as e:
                        print(f"Error in event loop handling for chunk {i+1}: {str(e)}")
                        raise
                    finally:
                        # Only close the loop if we created it
                        if loop and not loop.is_running():
                            try:
                                loop.close()
                                print(f"Closed event loop for chunk {i+1}")
                            except Exception as e:
                                print(f"Error closing event loop for chunk {i+1}: {str(e)}")

                    # Combine all samples
                    if all_samples:
                        samples = np.concatenate(all_samples)
                        sample_rate = sample_rate_container[0]  # Get the sample rate from the container

                        # Save the audio to a file
                        sf.write(chunk_path, samples, sample_rate)

                        # Calculate duration
                        duration = len(samples) / sample_rate

                        # Process the direct timing information
                        if all_timings:
                            print(f"Using direct timing information for chunk {i+1}: {len(all_timings)} words")
                            word_timings = self._process_direct_timings(all_timings, chunk)
                        else:
                            # Fallback to heuristic timing if no direct timings
                            print(f"No direct timing information for chunk {i+1}, using heuristic")
                            chunk_words = chunk.split()
                            word_timings = self._extract_word_timings(chunk_words, duration)
                    else:
                        # If no samples were collected, fall back to create method
                        raise Exception("No audio samples collected from stream")

                except Exception as stream_error:
                    print(f"Error using create_stream: {str(stream_error)}. Falling back to create method.")
                    # Fallback to the create method
                    samples, sample_rate = self.kokoro.create(
                        chunk, voice=voice, speed=speed, lang="en-us"
                    )

                    # Save the audio to a file
                    sf.write(chunk_path, samples, sample_rate)

                    # Calculate duration
                    duration = len(samples) / sample_rate

                    # Extract word timings using heuristic method
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
                duration = word_timings[-1]["end"] - time_offset if word_timings else 0
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

        # Store the complete word timings (both list and dictionary versions)
        self.word_timings_list = all_word_timings
        self.word_timings = self._convert_word_timings_to_dict(all_word_timings)

        print(f"Final word timings dictionary has {len(self.word_timings)} entries")

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

    def find_chunk_for_position(self, position: float) -> int:
        """
        Find the chunk index containing the given position.

        Args:
            position: Position in seconds.

        Returns:
            Chunk index or -1 if not found.
        """
        # Check if we have word timings
        if not self.word_timings:
            return -1

        # Find the word at the given position
        word_index = -1
        for i, timing in enumerate(self.word_timings):
            if timing["start"] <= position <= timing["end"]:
                word_index = i
                break

        if word_index == -1:
            # If not found, find the closest word before the position
            for i, timing in enumerate(self.word_timings):
                if timing["start"] > position:
                    if i > 0:
                        word_index = i - 1
                    break

        if word_index == -1:
            return -1

        # Calculate which chunk this word belongs to
        chunk_index = word_index // self.chunk_size
        return chunk_index

    def rewind_to_chunk(self, chunk_index: int) -> bool:
        """
        Rewind playback to the specified chunk.

        Args:
            chunk_index: The chunk index to rewind to.

        Returns:
            True if successful, False otherwise.
        """
        # Stop current playback
        self.pause_requested = True

        # Wait a bit for playback to stop
        time.sleep(0.2)

        # Clear the queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        # Find the chunk file
        chunk_files = [f for f in self.chunk_files if f.endswith('.wav')]
        if not chunk_files or chunk_index >= len(chunk_files):
            return False

        # Add the chunks from the specified index to the queue
        for i, chunk_file in enumerate(chunk_files[chunk_index:]):
            # Get the word timings for this chunk
            chunk_word_timings = []
            chunk_start = chunk_index * self.chunk_size
            chunk_end = min(chunk_start + self.chunk_size, len(self.word_timings))

            if chunk_start < len(self.word_timings):
                chunk_word_timings = self.word_timings[chunk_start:chunk_end]

            # Add to the queue
            self.audio_queue.put((chunk_index + i, len(chunk_files), chunk_file, chunk_word_timings))

        # Resume playback
        self.pause_requested = False
        return True

    def clear_all_cache(self):
        """
        Clear all cached audio files and reset state.
        This removes all files in the chunks directory and resets the engine state.
        """
        # Stop any ongoing processes
        self.stop_requested = True
        self.pause_requested = False
        self.synthesis_complete = False
        self.current_position = 0.0

        # Clear the queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        # Clear the chunk files list
        self.chunk_files = []

        # Remove all files in the chunks directory
        try:
            for file in os.listdir(self.chunks_dir):
                if file.endswith('.wav'):
                    file_path = os.path.join(self.chunks_dir, file)
                    try:
                        os.remove(file_path)
                        print(f"Removed cached file: {file_path}")
                    except Exception as e:
                        warnings.warn(f"Failed to remove file {file_path}: {str(e)}")
        except Exception as e:
            warnings.warn(f"Error clearing cache: {str(e)}")

        # Also clear the temp directory
        try:
            for file in os.listdir(self.temp_dir):
                if file.endswith('.wav') and os.path.isfile(os.path.join(self.temp_dir, file)):
                    file_path = os.path.join(self.temp_dir, file)
                    try:
                        os.remove(file_path)
                        print(f"Removed temp file: {file_path}")
                    except Exception as e:
                        warnings.warn(f"Failed to remove file {file_path}: {str(e)}")
        except Exception as e:
            warnings.warn(f"Error clearing temp directory: {str(e)}")

    def stop_all_tasks(self):
        """Stop all async tasks and clean up resources."""
        print("Stopping all TTS engine tasks")

        # Stop any ongoing processes
        self.stop_requested = True
        self.pause_requested = False

        # Stop any audio playback
        try:
            import sounddevice as sd
            sd.stop()
        except Exception as e:
            print(f"Error stopping sounddevice: {str(e)}")

        # Clear any queued audio
        if hasattr(self, 'audio_queue'):
            try:
                while not self.audio_queue.empty():
                    self.audio_queue.get_nowait()
            except Exception as e:
                print(f"Error clearing audio queue: {str(e)}")

        # Cancel any pending asyncio tasks
        try:
            # Try to get the current event loop safely
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running event loop, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Cancel all pending tasks
            try:
                for task in asyncio.all_tasks(loop):
                    if not task.done():
                        print(f"Cancelling task: {task.get_name()}")
                        task.cancel()
            except RuntimeError as e:
                print(f"Error getting tasks: {str(e)}")

            # Close the loop if we created it
            if loop.is_running():
                print("Event loop is running, not closing it")
            else:
                print("Closing event loop")
                loop.close()
        except Exception as e:
            print(f"Error handling asyncio tasks: {str(e)}")

    def unload_model(self):
        """Unload the model to free memory and clean up resources."""
        # First stop all tasks
        self.stop_all_tasks()

        # Set the model to None to free memory
        if self.kokoro is not None:
            print("Unloading Kokoro model")
            self.kokoro = None

        print("TTS model unloaded successfully")
