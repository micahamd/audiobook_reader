# Audiobook Reader

An application for playing audiobooks with text highlighting and editing capabilities.

## Features

- Import audio files (MP3, WAV) and transcribe them to text
- Import text files in various formats (TXT, MD, DOCX, PDF, etc.)
- Text-to-speech playback using Kokoro TTS model
- Word highlighting during playback
- Text editing when paused
- Customizable voice and speed settings
- State persistence

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

   Note: For full TTS functionality, you need the Kokoro model files.

   If you already have the Kokoro project, you can copy the model files:
   ```
   python copy_kokoro_models.py
   ```

   Or download them automatically:
   ```
   python download_models.py
   ```

   Or manually:
   1. Download [`kokoro-v0_19.onnx`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx) and place it in the `models/kokoro/` directory
   2. Download [`voices.json`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json) and place it in the `models/kokoro/` directory

   The application will still work without these model files, but will use a simple tone instead of actual speech synthesis.

## Usage

1. Run the application:
   ```
   python main.py
   ```

2. Import an audio file or text file using the toolbar buttons

3. Use the play/pause button to control playback

4. Edit text when paused (changes will be synthesized when you resume playback)

5. Adjust settings using the settings button in the toolbar

## Dependencies

- PyQt6: UI framework
- pydub: Audio processing
- markitdown-core: Text file conversion
- openai-whisper: Speech-to-text (optional)
- kokoro-onnx: Text-to-speech
- sounddevice: Audio playback
- soundfile: Audio file handling
- numpy: Numerical operations

## Testing

See the [Testing Guide](TESTING_GUIDE.md) for detailed instructions on testing the application.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx) for text-to-speech
- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [markitdown-core](https://github.com/markitdown/markitdown) for text file conversion
