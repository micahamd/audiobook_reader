# Audiobook Reader

A comprehensive application for playing audiobooks with text highlighting, editing capabilities, and advanced features for an enhanced reading experience.

## Features

- **Content Import**:
  - Import audio files (MP3, WAV) and transcribe them to text using Whisper
  - Import text files in various formats (TXT, MD, DOCX, PDF, etc.)
  - Automatic conversion to readable format

- **Text-to-Speech**:
  - High-quality TTS playback using Kokoro ONNX model
  - Streaming synthesis for immediate playback start
  - Adjustable playback speed
  - Progressive chunked processing for large texts

- **Reading Experience**:
  - Real-time word highlighting during playback
  - Text editing when paused with automatic re-synthesis
  - Pagination for efficient handling of large documents
  - Bookmarks for saving and returning to specific positions
  - Resume playback from exact position after pausing

- **Performance Optimizations**:
  - Background preprocessing of nearby pages
  - Lazy loading for large documents
  - Temporary caching of audio chunks
  - Efficient memory management

- **User Interface**:
  - Clean, intuitive PyQt6-based interface
  - Customizable voice and speed settings
  - Progress tracking and navigation controls
  - Bookmark management system

## Project Structure

```
audiobook_reader/
├── core/                   # Core functionality modules
│   ├── audio_processor.py  # Audio file processing
│   ├── background_processor.py # Background task management
│   ├── kokoro_onnx_engine.py # TTS engine implementation
│   ├── state_manager.py    # Application state persistence
│   ├── stt_engine.py       # Speech-to-text (Whisper) implementation
│   └── text_processor.py   # Text file processing
├── models/                 # Model files directory
│   └── kokoro/             # Kokoro TTS model files
│       ├── kokoro-v0_19.onnx # TTS model
│       └── voices.json     # Voice definitions
├── resources/              # Application resources
│   └── icons/              # UI icons
│       ├── add_bookmark.svg # Add bookmark icon
│       ├── app_icon.svg    # Application icon
│       ├── get_bookmark.svg # View bookmarks icon
│       ├── import_audio.svg # Import audio icon
│       ├── import_file.svg # Import file icon
│       ├── pause.svg       # Pause icon
│       ├── play.svg        # Play icon
│       └── settings.svg    # Settings icon
├── temp/                   # Temporary files directory
│   └── chunks/             # Audio chunk cache
├── ui/                     # User interface modules
│   ├── bookmarks_dialog.py # Bookmarks management dialog
│   ├── dialogs/            # Additional UI dialogs
│   │   ├── settings_dialog.py # Settings dialog
│   │   └── transcription_dialog.py # Transcription options dialog
│   ├── main_window.py      # Main application window
│   └── virtual_text_display.py # Optimized text display widget
├── utils/                  # Utility modules
│   ├── helpers.py          # Helper functions
│   └── threads.py          # Threading utilities
├── main.py                 # Application entry point
└── requirements.txt        # Python dependencies
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/audiobook_reader.git
   cd audiobook_reader
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up Kokoro TTS model files:

   **Option 1**: If you already have the Kokoro project, copy the model files:
   ```
   cp /path/to/kokoro-v0_19.onnx models/kokoro/
   cp /path/to/voices.json models/kokoro/
   ```

   **Option 2**: Download automatically:
   ```
   python download_kokoro_models.py
   ```

   **Option 3**: Download manually:
   1. Download [`kokoro-v0_19.onnx`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx) and place it in the `models/kokoro/` directory
   2. Download [`voices.json`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json) and place it in the `models/kokoro/` directory

   Note: The application will still work without these model files, but will use a simple tone instead of actual speech synthesis.

## Usage Guide

1. **Starting the Application**:
   ```
   python main.py
   ```

2. **Importing Content**:
   - Click the "Import Audio" button to import and transcribe audio files
   - Click the "Import File" button to import text files
   - The content will be automatically processed and displayed

3. **Navigation**:
   - Use the "Previous Page" and "Next Page" buttons to navigate through long documents
   - The current page and total pages are displayed between the navigation buttons

4. **Playback Controls**:
   - Click the Play button to start playback
   - Click the Pause button to pause playback
   - Use the slider to navigate to different positions in the audio
   - Current position and total duration are displayed next to the slider

5. **Text Editing**:
   - Text is read-only during playback
   - When paused, text becomes editable
   - Edited text will be automatically re-synthesized when playback resumes

6. **Bookmarks**:
   - Click "Add Bookmark" to save the current position
   - Click "Bookmarks" to view and manage saved bookmarks
   - Double-click a bookmark to jump to that position

7. **Settings**:
   - Click the Settings button to adjust voice, speed, and other options
   - Changes take effect immediately

8. **Cache Management**:
   - Click "Clear Cache" to remove temporary audio files and reset playback state

## Key Components and Implementation Details

### Text-to-Speech Engine (KokoroOnnxEngine)
- Implements streaming synthesis for immediate playback
- Processes text in chunks for efficient memory usage
- Maintains word timing information for highlighting
- Handles pause/resume with position tracking

### Background Processing
- Preprocesses nearby pages while reading the current page
- Manages a task queue with priority levels
- Provides callbacks for completed tasks

### Virtual Text Display
- Efficiently handles large documents
- Supports text highlighting during playback
- Maintains cursor position during scrolling

### Bookmark System
- Stores position, page, and text information
- Provides a dialog for managing bookmarks
- Supports navigation between bookmarks

## Development Roadmap

### Completed Features
- [x] Basic audio and text file import
- [x] Text-to-speech playback with Kokoro
- [x] Word highlighting during playback
- [x] Text editing when paused
- [x] Pagination for large documents
- [x] Background preprocessing
- [x] Bookmark system
- [x] Position tracking and resuming
- [x] Cursor position preservation

### Planned Enhancements
- [ ] Enhanced text formatting options
- [ ] Multiple voice support within a document
- [ ] Export synthesized audio
- [ ] Reading statistics and progress tracking
- [ ] Cloud synchronization of bookmarks and settings
- [ ] Mobile version

## Dependencies

- **UI Framework**:
  - PyQt6: Modern UI components and event handling

- **Audio Processing**:
  - pydub: Audio file manipulation
  - sounddevice: Real-time audio playback
  - numpy: Numerical operations for audio processing

- **Text Processing**:
  - markitdown-core: Text file format conversion

- **Machine Learning**:
  - openai-whisper: Speech-to-text transcription
  - kokoro-onnx: Neural text-to-speech synthesis

## Troubleshooting

### Common Issues

1. **Audio Playback Problems**:
   - Ensure your audio device is properly configured
   - Check that the Kokoro model files are correctly installed
   - Try clearing the cache and restarting the application

2. **Text Import Issues**:
   - Verify the file format is supported
   - Check file encoding (UTF-8 is recommended)
   - For large files, allow more time for processing

3. **Performance Concerns**:
   - Adjust the page size in settings for your hardware
   - Close other memory-intensive applications
   - Consider using SSD storage for better cache performance

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx) for text-to-speech
- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [markitdown-core](https://github.com/markitdown/markitdown) for text file conversion
