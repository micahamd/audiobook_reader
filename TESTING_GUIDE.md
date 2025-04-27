# Audiobook Reader Testing Guide

This guide provides instructions for testing the various features of the Audiobook Reader application.

## Prerequisites

Before running the tests, make sure you have:

1. Installed all dependencies:
   ```
   pip install -r requirements.txt
   ```

   For full TTS functionality, you need the Kokoro model files.

   If you already have the Kokoro project, copy the model files:
   ```
   cp /path/to/kokoro-v0_19.onnx models/kokoro/
   cp /path/to/voices.json models/kokoro/
   ```

   Or download them automatically:
   ```
   python download_kokoro_models.py
   ```

   Or manually download:
   1. [`kokoro-v0_19.onnx`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx) to the `models/kokoro/` directory
   2. [`voices.json`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json) to the `models/kokoro/` directory

2. Set up the application:
   ```
   python main.py
   ```

### Testing with Limited Dependencies

The application includes fallback functionality when Kokoro model files are not available. To test this:

1. Create a virtual environment with the basic dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the application without downloading the Kokoro model files
3. Verify that it works with the fallback TTS (simple tone)

### Testing with Model Files in a Different Location

If you have the Kokoro model files in a different location, you can modify the code to use them:

1. Open `core/kokoro_onnx_engine.py`
2. Update the `__init__` method to use your model file paths:
   ```python
   self.model_path = "/path/to/kokoro-v0_19.onnx"
   self.voices_path = "/path/to/voices.json"
   ```
3. Run the application and verify that it uses the specified model files

## Unit Tests

The application includes unit tests for core components. Run them with:

```
python -m unittest discover tests
```

To run specific tests:

```
python -m unittest tests/test_text_processor.py
python -m unittest tests/test_tts_engine.py
```

## Feature Testing

### 1. Audio Import and Transcription

**Test Steps:**
1. Launch the application
2. Click "Import Audio" in the toolbar
3. Select an MP3 or WAV file
4. In the transcription dialog, select "base" model and click "OK"
5. Wait for transcription to complete

**Expected Results:**
- Transcribed text appears in the text display
- Audio is synthesized and ready for playback

### 2. File Import

**Test Steps:**
1. Launch the application
2. Click "Import File" in the toolbar
3. Select a text file (TXT, MD, DOCX, etc.)
4. Wait for processing to complete

**Expected Results:**
- File content appears in the text display
- Audio is synthesized and ready for playback

### 3. Text-to-Speech Playback

**Test Steps:**
1. Import a file or audio as described above
2. Click the play button
3. Listen to the audio
4. Click the pause button
5. Click play again to resume

**Expected Results:**
- Audio plays when play button is clicked
- Play button icon changes to pause icon
- Audio pauses when pause button is clicked
- Audio resumes from the paused position

### 4. Text Highlighting

**Test Steps:**
1. Import a file or audio as described above
2. Click the play button
3. Observe the text display

**Expected Results:**
- Words are highlighted in sync with the audio playback
- Highlighted words are visible in the text display (may need to scroll)

### 5. Text Editing

**Test Steps:**
1. Import a file or audio as described above
2. Click the play button, then pause
3. Edit some text in the text display
4. Click play again

**Expected Results:**
- Text becomes editable when paused
- After editing, clicking play triggers re-synthesis
- New audio reflects the edited text

### 6. Chunked Text Processing

**Test Steps:**
1. Import a large text file (>1000 words)
2. Observe the processing time
3. Play the audio and observe performance

**Expected Results:**
- Large files are processed in chunks
- Playback is smooth even for large files
- Word highlighting works correctly across chunk boundaries

### 7. Settings

**Test Steps:**
1. Click "Settings" in the toolbar
2. Change voice and speed settings
3. Click "OK"
4. Play the audio

**Expected Results:**
- Settings dialog opens
- After changing settings, audio is re-synthesized
- New audio reflects the changed settings

### 8. State Persistence

**Test Steps:**
1. Import a file and play to a specific position
2. Close the application
3. Reopen the application

**Expected Results:**
- The same file is loaded
- Settings are preserved
- Position is remembered (may need to implement position saving)

## Performance Testing

### Large File Handling

**Test Steps:**
1. Import a very large text file (e.g., a novel)
2. Observe memory usage and processing time
3. Play the audio and navigate through the text

**Expected Results:**
- Application remains responsive
- Memory usage remains reasonable
- Playback is smooth

### Continuous Playback

**Test Steps:**
1. Import a medium-sized file
2. Start playback and let it run for at least 10 minutes

**Expected Results:**
- Playback continues without interruption
- Word highlighting remains in sync
- Application remains responsive

## Bug Reporting

If you encounter any issues during testing, please report them with:

1. Steps to reproduce the issue
2. Expected behavior
3. Actual behavior
4. Screenshots or error messages if applicable
5. System information (OS, Python version, etc.)

## Troubleshooting

### Common Issues

1. **Audio not playing:**
   - Check audio output device settings
   - Ensure audio files are generated correctly in the temp directory

2. **Slow processing:**
   - Check CPU and memory usage
   - Consider reducing chunk size for better performance

3. **Word highlighting issues:**
   - Check text format and encoding
   - Ensure word timings are calculated correctly
