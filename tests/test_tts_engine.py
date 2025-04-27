"""
Tests for the TTS engine module.
"""

import os
import unittest
import tempfile
from pathlib import Path

import numpy as np

from core.tts_engine import TTSEngine, TextChunk


class TestTTSEngine(unittest.TestCase):
    """Tests for the TTSEngine class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tts_engine = TTSEngine(temp_dir=self.temp_dir.name)
    
    def tearDown(self):
        """Clean up the test environment."""
        self.temp_dir.cleanup()
    
    def test_split_text_into_chunks(self):
        """Test splitting text into chunks."""
        # Create a test text with more than CHUNK_SIZE words
        words = ["word" + str(i) for i in range(250)]
        test_text = " ".join(words)
        
        # Split into chunks
        chunks = self.tts_engine.split_text_into_chunks(test_text)
        
        # Check that we have the expected number of chunks
        expected_chunks = (len(words) + self.tts_engine.CHUNK_SIZE - 1) // self.tts_engine.CHUNK_SIZE
        self.assertEqual(len(chunks), expected_chunks)
        
        # Check that each chunk has the expected number of words
        for i, chunk in enumerate(chunks):
            if i < len(chunks) - 1:
                # All chunks except the last one should have CHUNK_SIZE words
                chunk_words = chunk.text.split()
                self.assertEqual(len(chunk_words), self.tts_engine.CHUNK_SIZE)
            else:
                # The last chunk might have fewer words
                chunk_words = chunk.text.split()
                self.assertLessEqual(len(chunk_words), self.tts_engine.CHUNK_SIZE)
    
    def test_extract_word_timings(self):
        """Test extracting word timings."""
        # Create a test text
        test_text = "This is a test sentence with multiple words."
        
        # Extract timings
        duration = 5.0  # 5 seconds
        timings = self.tts_engine._extract_word_timings(test_text, duration)
        
        # Check that we have the expected number of timings
        words = test_text.split()
        self.assertEqual(len(timings), len(words))
        
        # Check that the timings are within the duration
        for timing in timings:
            self.assertGreaterEqual(timing["start"], 0)
            self.assertLessEqual(timing["end"], duration)
            self.assertLess(timing["start"], timing["end"])
    
    def test_text_chunk(self):
        """Test the TextChunk class."""
        # Create a test chunk
        chunk = TextChunk("This is a test chunk.", 1)
        
        # Check initial state
        self.assertEqual(chunk.text, "This is a test chunk.")
        self.assertEqual(chunk.chunk_id, 1)
        self.assertFalse(chunk.processed)
        
        # Set audio data
        audio_path = "test.wav"
        word_timings = [
            {"word": "This", "start": 0.0, "end": 0.5},
            {"word": "is", "start": 0.5, "end": 0.7},
            {"word": "a", "start": 0.7, "end": 0.8},
            {"word": "test", "start": 0.8, "end": 1.2},
            {"word": "chunk.", "start": 1.2, "end": 1.5}
        ]
        duration = 1.5
        
        chunk.set_audio_data(audio_path, word_timings, duration)
        
        # Check updated state
        self.assertEqual(chunk.audio_path, audio_path)
        self.assertEqual(chunk.word_timings, word_timings)
        self.assertEqual(chunk.duration, duration)
        self.assertTrue(chunk.processed)
        
        # Test timing adjustment
        start_time = 10.0
        chunk.adjust_timings(start_time)
        
        # Check adjusted timings
        self.assertEqual(chunk.start_time, start_time)
        for i, timing in enumerate(chunk.word_timings):
            self.assertEqual(timing["start"], word_timings[i]["start"] + start_time)
            self.assertEqual(timing["end"], word_timings[i]["end"] + start_time)


if __name__ == '__main__':
    unittest.main()
