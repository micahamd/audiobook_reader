"""
Main window module for the Audiobook Reader application.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union

from PyQt6.QtCore import QTimer, Qt, pyqtSlot, QSize, QUrl
from PyQt6.QtGui import QAction, QFont, QTextCursor, QTextCharFormat, QColor, QIcon
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QToolBar, QFileDialog,
    QLabel, QSlider, QComboBox, QMessageBox, QProgressBar
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput, QAudioFormat, QMediaDevices


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

from core.audio_processor import AudioProcessor
from core.text_processor import TextProcessor
from core.stt_engine import STTEngine
from core.kokoro_onnx_engine import KokoroOnnxEngine
from core.state_manager import StateManager
from ui.dialogs.transcription_dialog import TranscriptionDialog
from ui.dialogs.settings_dialog import SettingsDialog
from utils.helpers import (
    validate_file_path, get_supported_audio_extensions,
    get_supported_text_extensions, format_time
)
from utils.threads import ThreadManager


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        """Initialize the main window."""
        super().__init__()

        # Initialize core components
        self.audio_processor = AudioProcessor()
        self.text_processor = TextProcessor()
        self.stt_engine = STTEngine()
        self.tts_engine = KokoroOnnxEngine()
        self.state_manager = StateManager()
        self.thread_manager = ThreadManager()

        # Initialize bookmark
        self.bookmark_position = 0.0
        self.bookmark_text_position = 0

        # Initialize media player
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)

        # Initialize UI state
        self.current_file_path = None
        self.current_audio_path = None
        self.current_text = ""
        self.word_timings = []
        self.is_playing = False
        self.highlight_timer = QTimer()
        self.highlight_timer.timeout.connect(self.update_highlight)

        # Timer for delayed re-synthesis after text edits
        self.edit_timer = QTimer()
        self.edit_timer.setSingleShot(True)
        self.edit_timer.timeout.connect(self.handle_text_edit)
        self.text_edited = False

        # Set up the UI
        self.setup_ui()

        # Load saved state
        self.load_state()

    def setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("Audiobook Reader")
        self.resize(800, 600)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create text display
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)  # Initially read-only, will be made editable when paused
        self.text_display.setFont(QFont("Arial", 12))
        self.text_display.textChanged.connect(self.on_text_changed)
        main_layout.addWidget(self.text_display)

        # Create playback controls
        controls_layout = QHBoxLayout()

        # Load icons
        play_icon = QIcon(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'icons', 'play.svg'))
        pause_icon = QIcon(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'icons', 'pause.svg'))

        # Create play button with icon
        self.play_button = QPushButton()
        self.play_button.setIcon(play_icon)
        self.play_button.setIconSize(QSize(24, 24))
        self.play_button.setToolTip("Play/Pause audio")
        self.play_button.clicked.connect(self.toggle_playback)
        controls_layout.addWidget(self.play_button)

        # Store icons for later use
        self.play_icon = play_icon
        self.pause_icon = pause_icon

        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 100)
        self.position_slider.setToolTip("Playback position")
        self.position_slider.sliderMoved.connect(self.set_position)
        controls_layout.addWidget(self.position_slider)

        self.time_label = QLabel("00:00 / 00:00")
        controls_layout.addWidget(self.time_label)

        main_layout.addLayout(controls_layout)

        # Create toolbar
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        # Load toolbar icons
        import_audio_icon = QIcon(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'icons', 'import_audio.svg'))
        import_file_icon = QIcon(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'icons', 'import_file.svg'))
        settings_icon = QIcon(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'icons', 'settings.svg'))
        app_icon = QIcon(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'icons', 'app_icon.svg'))

        # Set application icon
        self.setWindowIcon(app_icon)

        # Add actions to toolbar with icons
        import_audio_action = QAction(import_audio_icon, "Import Audio", self)
        import_audio_action.setToolTip("Import audio file (MP3, WAV) for transcription")
        import_audio_action.triggered.connect(self.import_audio)
        toolbar.addAction(import_audio_action)

        import_file_action = QAction(import_file_icon, "Import File", self)
        import_file_action.setToolTip("Import text file (TXT, MD, DOCX, PDF, etc.)")
        import_file_action.triggered.connect(self.import_file)
        toolbar.addAction(import_file_action)

        # Add bookmark actions
        self.add_bookmark_action = QAction("Add Bookmark", self)
        self.add_bookmark_action.setToolTip("Save current position as bookmark")
        self.add_bookmark_action.triggered.connect(self.add_bookmark)
        toolbar.addAction(self.add_bookmark_action)

        self.goto_bookmark_action = QAction("Go to Bookmark", self)
        self.goto_bookmark_action.setToolTip("Jump to saved bookmark position")
        self.goto_bookmark_action.triggered.connect(self.goto_bookmark)
        toolbar.addAction(self.goto_bookmark_action)

        settings_action = QAction(settings_icon, "Settings", self)
        settings_action.setToolTip("Configure voice, speed, and other settings")
        settings_action.triggered.connect(self.show_settings)
        toolbar.addAction(settings_action)

        # Set up status bar
        self.status_bar = self.statusBar()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        # Connect media player signals
        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.durationChanged.connect(self.duration_changed)
        self.media_player.playbackStateChanged.connect(self.playback_state_changed)

    def load_state(self):
        """Load saved state."""
        # Load window geometry
        window_size = self.state_manager.get("window_size")
        window_position = self.state_manager.get("window_position")

        if window_size:
            self.resize(window_size[0], window_size[1])

        if window_position:
            self.move(window_position[0], window_position[1])

        # Load last file if it exists
        last_file = self.state_manager.get("last_file")
        if last_file and validate_file_path(last_file):
            self.load_file(last_file)

    def save_state(self):
        """Save current state."""
        # Save window geometry
        self.state_manager.set("window_size", [self.width(), self.height()])
        self.state_manager.set("window_position", [self.x(), self.y()])

        # Save current file
        if self.current_file_path:
            self.state_manager.set("last_file", self.current_file_path)

        # Save current position
        if self.media_player.position() > 0:
            self.state_manager.set("last_position", self.media_player.position())

        # Save state to disk
        self.state_manager.save_state()

    def closeEvent(self, event):
        """Handle window close event."""
        self.save_state()
        event.accept()

    def import_audio(self):
        """Import an audio file."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Import Audio File",
            "",
            f"Audio Files ({' '.join(['*' + ext for ext in get_supported_audio_extensions()])})"
        )

        if file_path:
            self.load_audio(file_path)

    def import_file(self):
        """Import a text file."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Import Text File",
            "",
            f"Text Files ({' '.join(['*' + ext for ext in get_supported_text_extensions()])})"
        )

        if file_path:
            self.load_file(file_path)

    def load_audio(self, file_path: str):
        """
        Load an audio file.

        Args:
            file_path: Path to the audio file.
        """
        try:
            # Show progress
            self.status_bar.showMessage(f"Loading audio: {os.path.basename(file_path)}")
            self.progress_bar.setVisible(True)

            # Load the audio file
            audio, format_str = self.audio_processor.load_audio(file_path)

            # Ask if the user wants to transcribe
            dialog = TranscriptionDialog(self)
            if dialog.exec():
                # User wants to transcribe
                self.transcribe_audio(file_path)
            else:
                # User just wants to play the audio
                self.current_file_path = file_path
                self.current_audio_path = file_path
                self.current_text = ""
                self.text_display.clear()
                # Convert the file path to a QUrl
                file_url = QUrl.fromLocalFile(file_path)
                self.media_player.setSource(file_url)
                self.status_bar.showMessage(f"Loaded audio: {os.path.basename(file_path)}")

            self.progress_bar.setVisible(False)

        except Exception as e:
            self.progress_bar.setVisible(False)
            self.status_bar.showMessage("")
            QMessageBox.critical(self, "Error", f"Failed to load audio: {str(e)}")

    def transcribe_audio(self, file_path: str):
        """
        Transcribe an audio file.

        Args:
            file_path: Path to the audio file.
        """
        def transcribe_task(progress_callback):
            # Convert to WAV if needed
            audio, _ = self.audio_processor.load_audio(file_path)
            wav_path = self.audio_processor.convert_to_wav(audio)
            progress_callback.emit(20)

            # Transcribe
            result = self.stt_engine.transcribe(wav_path)
            progress_callback.emit(90)

            return result

        # Start transcription in a background thread
        worker = self.thread_manager.start_worker(transcribe_task)

        # Connect signals
        worker.signals.progress.connect(self.progress_bar.setValue)
        worker.signals.result.connect(self.handle_transcription_result)
        worker.signals.error.connect(self.handle_worker_error)
        worker.signals.started.connect(lambda: self.status_bar.showMessage("Transcribing audio..."))

    def handle_transcription_result(self, result):
        """
        Handle transcription result.

        Args:
            result: The transcription result.
        """
        self.current_text = result["text"]
        self.text_display.setPlainText(self.current_text)
        self.status_bar.showMessage("Transcription complete")
        self.progress_bar.setVisible(False)

        # Synthesize speech
        self.synthesize_speech()

    def load_file(self, file_path: str):
        """
        Load a text file.

        Args:
            file_path: Path to the text file.
        """
        try:
            # Show progress
            self.status_bar.showMessage(f"Loading file: {os.path.basename(file_path)}")
            self.progress_bar.setVisible(True)

            # Load the file
            content, format_str = self.text_processor.load_file(file_path)

            # Update UI
            self.current_file_path = file_path
            self.current_text = content
            self.text_display.setPlainText(content)

            # Make text editable immediately
            self.text_display.setReadOnly(False)
            self.status_bar.showMessage("Text is editable. Edit as needed, then press Play to synthesize.")

            # Synthesize speech
            self.synthesize_speech()

            self.status_bar.showMessage(f"Loaded file: {os.path.basename(file_path)}")
            self.progress_bar.setVisible(False)

        except Exception as e:
            self.progress_bar.setVisible(False)
            self.status_bar.showMessage("")
            QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")

    def synthesize_speech(self):
        """Synthesize speech from the current text."""
        if not self.current_text:
            return

        # Get TTS settings
        tts_settings = self.state_manager.get("tts_settings", {})
        voice = tts_settings.get("voice", "af_sarah")
        speed = tts_settings.get("speed", 1.0)

        # Stop any existing playback
        if self.is_playing:
            self.media_player.stop()

        # Stop any existing TTS processes
        self.tts_engine.stop_requested = True

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage("Synthesizing speech... (will start playing automatically)")

        # Define callback for chunk progress
        def chunk_callback(chunk_index, total_chunks, audio_path, word_timings):
            # Update progress
            progress = int((chunk_index + 1) / total_chunks * 100)
            self.progress_bar.setValue(progress)

            # If this is the first chunk, start playback
            if chunk_index == 0:
                self.handle_first_chunk(audio_path, word_timings)

        try:
            # Start progressive synthesis and playback
            self.synthesis_thread, self.playback_thread = self.tts_engine.synthesize_and_play_progressively(
                self.current_text,
                voice=voice,
                speed=speed,
                callback=chunk_callback
            )

            # Update UI state
            self.is_playing = True
            self.play_button.setIcon(self.pause_icon)
            self.text_display.setReadOnly(True)
            self.highlight_timer.start(100)  # Update highlight every 100ms

        except Exception as e:
            # Handle errors
            self.progress_bar.setVisible(False)
            self.status_bar.showMessage(f"Error: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to synthesize speech: {str(e)}")

            # Fallback to non-progressive synthesis
            self.fallback_synthesis(voice, speed)

    def handle_first_chunk(self, audio_path, word_timings):
        """
        Handle the first audio chunk.

        Args:
            audio_path: Path to the audio file.
            word_timings: Word timing information.
        """
        print(f"First chunk ready. Audio path: {audio_path}")

        # Store the audio path and word timings
        self.current_audio_path = audio_path
        self.word_timings = word_timings

        # Update status
        self.status_bar.showMessage("Starting playback...")

    def fallback_synthesis(self, voice, speed):
        """
        Fallback to non-progressive synthesis.

        Args:
            voice: The voice to use.
            speed: The speed factor.
        """
        def synthesize_task(progress_callback):
            """
            Task for synthesizing speech.

            Args:
                progress_callback: Callback for reporting progress.

            Returns:
                Tuple of (audio_path, word_timings).
            """
            progress_callback.emit(10)

            # Prepare text
            progress_callback.emit(20)

            # Initialize TTS engine
            progress_callback.emit(30)

            # Synthesize speech
            progress_callback.emit(50)

            try:
                # Synthesize using the chunked approach
                audio_path, word_timings = self.tts_engine.synthesize(
                    self.current_text,
                    voice=voice,
                    speed=speed
                )

                progress_callback.emit(90)

                return audio_path, word_timings
            except ImportError as e:
                # Handle the case when TTS dependencies are not available
                self.status_bar.showMessage(f"Warning: {str(e)}")
                QMessageBox.warning(
                    self,
                    "TTS Dependencies Missing",
                    "Some TTS dependencies are missing. Using a simple tone instead of speech synthesis.\n\n"
                    "To enable full TTS functionality, please install the required packages:\n"
                    "pip install kokoro-onnx"
                )

                # Use dummy synthesis
                audio_path, word_timings = self.tts_engine._dummy_synthesize(self.current_text, speed)

                progress_callback.emit(90)

                return audio_path, word_timings

        # Start synthesis in a background thread
        worker = self.thread_manager.start_worker(synthesize_task)

        # Connect signals
        worker.signals.progress.connect(self.progress_bar.setValue)
        worker.signals.result.connect(self.handle_synthesis_result)
        worker.signals.error.connect(self.handle_worker_error)
        worker.signals.started.connect(lambda: self.status_bar.showMessage("Synthesizing speech..."))
        self.progress_bar.setVisible(True)

    def handle_synthesis_result(self, result):
        """
        Handle synthesis result.

        Args:
            result: Tuple of (audio_path, word_timings).
        """
        audio_path, word_timings = result

        print(f"Synthesis complete. Audio path: {audio_path}")
        print(f"Word timings count: {len(word_timings)}")

        # Check if the audio file exists
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file does not exist at {audio_path}")
            self.status_bar.showMessage("Error: Audio file not found")
            return

        self.current_audio_path = audio_path
        self.word_timings = word_timings

        # Load the audio into the media player
        print(f"Setting media source to: {audio_path}")
        # Convert the file path to a QUrl
        file_url = QUrl.fromLocalFile(audio_path)
        print(f"File URL: {file_url.toString()}")
        self.media_player.setSource(file_url)

        # Check if the source was set correctly
        if self.media_player.source().isEmpty():
            print("Warning: Media source is empty after setting")
            self.status_bar.showMessage("Error: Could not load audio file")
        else:
            print(f"Media source set to: {self.media_player.source().toString()}")
            # Make sure text is editable after synthesis
            self.text_display.setReadOnly(False)
            self.status_bar.showMessage("Synthesis complete. Text is editable. Press play to start.")

        self.progress_bar.setVisible(False)

    def handle_worker_error(self, error_info):
        """
        Handle worker error.

        Args:
            error_info: Tuple of (exception, traceback, thread).
        """
        exception = error_info[0]
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("")
        QMessageBox.critical(self, "Error", f"An error occurred: {str(exception)}")

    # This method is replaced by the one below that handles text edits

    def playback_state_changed(self, state):
        """
        Handle playback state changes.

        Args:
            state: The new playback state.
        """
        print(f"Playback state changed to: {state}")

        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.is_playing = True
            self.play_button.setIcon(self.pause_icon)
            self.text_display.setReadOnly(True)
            self.highlight_timer.start(100)  # Update highlight every 100ms
            self.status_bar.showMessage("Playing audio...")
        else:
            self.is_playing = False
            self.play_button.setIcon(self.play_icon)
            self.text_display.setReadOnly(False)  # Make text editable when paused
            self.highlight_timer.stop()

            if state == QMediaPlayer.PlaybackState.PausedState:
                self.status_bar.showMessage("Paused. Text is now editable.")
            elif state == QMediaPlayer.PlaybackState.StoppedState:
                self.status_bar.showMessage("Stopped. Text is now editable.")

    def position_changed(self, position):
        """
        Handle media player position changes.

        Args:
            position: The new position in milliseconds.
        """
        # Update position slider
        if self.media_player.duration() > 0:
            self.position_slider.setValue(int(position / self.media_player.duration() * 100))

        # Update time label
        current_time = format_time(position / 1000)
        total_time = format_time(self.media_player.duration() / 1000)
        self.time_label.setText(f"{current_time} / {total_time}")

    def duration_changed(self, duration):
        """
        Handle media player duration changes.

        Args:
            duration: The new duration in milliseconds.
        """
        # Update time label
        current_time = format_time(self.media_player.position() / 1000)
        total_time = format_time(duration / 1000)
        self.time_label.setText(f"{current_time} / {total_time}")

    def set_position(self, position):
        """
        Set the media player position.

        Args:
            position: The position as a percentage (0-100).
        """
        if self.media_player.duration() > 0:
            new_position = int(position / 100 * self.media_player.duration())
            self.media_player.setPosition(new_position)

    def update_highlight(self):
        """Update the text highlighting based on the current playback position."""
        if not self.word_timings or not self.is_playing:
            return

        # Get current time - either from media player or from TTS engine
        if hasattr(self, 'synthesis_thread') and self.synthesis_thread and self.synthesis_thread.is_alive():
            # For progressive playback, use the TTS engine's current position
            current_time = self.tts_engine.current_position
        else:
            # For media player playback, convert from milliseconds to seconds
            current_time = self.media_player.position() / 1000

        # Get the current word from the TTS engine
        current_word = self.tts_engine.get_word_at_position(current_time)

        if current_word:
            # Clear previous highlighting
            cursor = self.text_display.textCursor()
            cursor.select(QTextCursor.SelectionType.Document)
            cursor.setCharFormat(QTextCharFormat())

            # Get the word to highlight
            word = current_word["word"]

            # Find the word in the text
            cursor = self.text_display.textCursor()
            cursor.setPosition(0)

            # Create a format for highlighting
            highlight_format = QTextCharFormat()
            highlight_format.setBackground(QColor(255, 255, 0, 100))  # Light yellow
            highlight_format.setForeground(QColor(0, 0, 0))  # Black text

            # Find and highlight the word
            found = False
            while not found:
                found = cursor.movePosition(
                    QTextCursor.MoveOperation.NextWord,
                    QTextCursor.MoveMode.KeepAnchor
                )
                if not found:
                    break

                selected_text = cursor.selectedText().strip()
                if selected_text.lower() == word.lower():
                    cursor.setCharFormat(highlight_format)
                    # Ensure the highlighted word is visible
                    self.text_display.ensureCursorVisible()
                    break

                # Reset selection and move to next word
                cursor.clearSelection()

    def on_text_changed(self):
        """Handle text changes in the text display."""
        if not self.is_playing and self.current_text:
            # Mark that text has been edited
            self.text_edited = True
            print("Text changed. Marked for re-synthesis.")

            # Start the timer to trigger re-synthesis after a delay
            self.edit_timer.start(2000)  # 2 seconds delay

    def handle_text_edit(self):
        """Handle text edit after the delay timer expires."""
        if self.text_edited:
            # Get the updated text
            new_text = self.text_display.toPlainText()
            print(f"Handling text edit. New text length: {len(new_text)}")

            # Only re-synthesize if the text has actually changed
            if new_text != self.current_text:
                self.current_text = new_text
                print("Text content changed. Will re-synthesize on play.")
                self.status_bar.showMessage("Text edited. Will re-synthesize on play.")

                # Stop any existing progressive playback
                if hasattr(self, 'synthesis_thread') and self.synthesis_thread and self.synthesis_thread.is_alive():
                    self.tts_engine.stop_requested = True
                    # Wait for the thread to finish
                    self.synthesis_thread.join(0.5)

                # Clear the current audio path to force a complete re-synthesis
                self.current_audio_path = None

    def toggle_playback(self):
        """Toggle playback between play and pause."""
        print(f"Toggle playback called. Current audio path: {self.current_audio_path}")
        print(f"Is playing: {self.is_playing}, Text edited: {self.text_edited}")

        if not self.current_text:
            self.status_bar.showMessage("No text available. Please import a file first.")
            return

        if self.is_playing:
            print("Pausing playback")
            # Toggle pause in the TTS engine for progressive playback
            self.tts_engine.toggle_pause()
            # Also pause the media player for non-progressive playback
            self.media_player.pause()
            # Update UI
            self.play_button.setIcon(self.play_icon)
            self.text_display.setReadOnly(False)
            self.highlight_timer.stop()
            self.is_playing = False
            self.status_bar.showMessage("Paused. Text is now editable.")
        else:
            # Check if text was edited and needs re-synthesis
            if self.text_edited:
                print("Text was edited, re-synthesizing")
                self.text_edited = False
                self.synthesize_speech()
            else:
                print("Starting playback")
                # If we're using progressive playback
                if hasattr(self, 'synthesis_thread') and self.synthesis_thread and self.synthesis_thread.is_alive():
                    # Resume the paused playback
                    self.tts_engine.toggle_pause()
                    # Update UI
                    self.play_button.setIcon(self.pause_icon)
                    self.text_display.setReadOnly(True)
                    self.highlight_timer.start(100)
                    self.is_playing = True
                    self.status_bar.showMessage("Resuming playback...")
                else:
                    # Start new synthesis if no current playback
                    if not self.current_audio_path:
                        self.synthesize_speech()
                    else:
                        # Use the media player for non-progressive playback
                        self.media_player.play()

                        # Check if playback actually started
                        if self.media_player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
                            print(f"Playback failed to start. Media player state: {self.media_player.playbackState()}")
                            print(f"Media player error: {self.media_player.error()}")
                            self.status_bar.showMessage(f"Playback error: {self.media_player.errorString()}")
                        else:
                            print("Playback started successfully")

    def add_bookmark(self):
        """Save the current position as a bookmark."""
        if not self.current_text:
            self.status_bar.showMessage("No text loaded. Cannot add bookmark.")
            return

        # Get current position
        if hasattr(self, 'synthesis_thread') and self.synthesis_thread and self.synthesis_thread.is_alive():
            # For progressive playback, use the TTS engine's current position
            self.bookmark_position = self.tts_engine.current_position
        else:
            # For media player playback, convert from milliseconds to seconds
            self.bookmark_position = self.media_player.position() / 1000

        # Get current text cursor position
        self.bookmark_text_position = self.text_display.textCursor().position()

        # Save to state manager
        self.state_manager.set("bookmark", {
            "position": self.bookmark_position,
            "text_position": self.bookmark_text_position,
            "file_path": self.current_file_path,
            "text": self.current_text[:100] + "..." if len(self.current_text) > 100 else self.current_text
        })

        # Update UI
        self.status_bar.showMessage(f"Bookmark added at position {format_time(self.bookmark_position)}")

    def goto_bookmark(self):
        """Jump to the saved bookmark position."""
        # Get bookmark from state manager
        bookmark = self.state_manager.get("bookmark")
        if not bookmark:
            self.status_bar.showMessage("No bookmark saved.")
            return

        # Check if the current file matches the bookmarked file
        if self.current_file_path != bookmark.get("file_path"):
            # Ask if the user wants to load the bookmarked file
            response = QMessageBox.question(
                self,
                "Load Bookmarked File",
                f"The bookmark is for a different file. Do you want to load it?\n\n{bookmark.get('file_path')}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if response == QMessageBox.StandardButton.Yes:
                # Load the file
                self.load_file(bookmark.get("file_path"))
            else:
                return

        # Set text cursor position
        cursor = self.text_display.textCursor()
        cursor.setPosition(bookmark.get("text_position", 0))
        self.text_display.setTextCursor(cursor)
        self.text_display.ensureCursorVisible()

        # Set media position
        if self.current_audio_path:
            position_ms = int(bookmark.get("position", 0) * 1000)
            self.media_player.setPosition(position_ms)

        # Update UI
        self.status_bar.showMessage(f"Jumped to bookmark at {format_time(bookmark.get('position', 0))}")

    def show_settings(self):
        """Show the settings dialog."""
        dialog = SettingsDialog(self.state_manager, self)
        if dialog.exec():
            # Settings were changed, update as needed
            self.synthesize_speech()
