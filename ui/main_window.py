"""
Main window module for the Audiobook Reader application.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union

from PyQt6.QtCore import QTimer, Qt, pyqtSlot, QSize, QUrl
from PyQt6.QtGui import QAction, QFont, QTextCursor, QTextCharFormat, QColor, QIcon, QTextDocument
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QToolBar, QFileDialog,
    QLabel, QSlider, QComboBox, QMessageBox, QProgressBar
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput, QAudioFormat, QMediaDevices

from ui.virtual_text_display import VirtualTextDisplay
from core.background_processor import BackgroundProcessor


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
from ui.bookmarks_dialog import BookmarksDialog
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

        # Initialize paging system
        self.full_text = ""  # The complete text of the book
        self.pages = []  # List of text pages
        self.current_page_index = 0
        self.page_size = 5000  # Characters per page (adjustable)

        # Initialize background processor
        self.background_processor = BackgroundProcessor()

        # Track pre-processed pages
        self.preprocessed_pages = set()

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
        self.auto_play = False  # Don't auto-play when loading files
        self.last_playback_position = 0  # Track the last playback position
        self.last_cursor_position = 0  # Track the last cursor position
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

        # Create virtual text display for efficient handling of large texts
        self.text_display = VirtualTextDisplay()
        self.text_display.setReadOnly(True)  # Initially read-only, will be made editable when paused
        self.text_display.setFont(QFont("Arial", 12))
        self.text_display.text_edited.connect(self.on_text_changed)
        self.text_display.content_changed.connect(self.on_content_changed)
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

        # Create page navigation controls
        page_layout = QHBoxLayout()

        self.prev_page_button = QPushButton("← Previous Page")
        self.prev_page_button.setToolTip("Go to previous page")
        self.prev_page_button.clicked.connect(self.go_to_previous_page)
        page_layout.addWidget(self.prev_page_button)

        self.page_label = QLabel("Page 1 of 1")
        self.page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        page_layout.addWidget(self.page_label)

        self.next_page_button = QPushButton("Next Page →")
        self.next_page_button.setToolTip("Go to next page")
        self.next_page_button.clicked.connect(self.go_to_next_page)
        page_layout.addWidget(self.next_page_button)

        main_layout.addLayout(page_layout)

        # Create toolbar
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        # Load toolbar icons
        import_audio_icon = QIcon(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'icons', 'import_audio.svg'))
        import_file_icon = QIcon(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'icons', 'import_file.svg'))
        settings_icon = QIcon(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'icons', 'settings.svg'))
        app_icon = QIcon(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'icons', 'app_icon.svg'))
        add_bookmark_icon = QIcon(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'icons', 'add_bookmark.svg'))
        get_bookmark_icon = QIcon(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'icons', 'get_bookmark.svg'))

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
        self.add_bookmark_action = QAction(add_bookmark_icon, "Add Bookmark", self)
        self.add_bookmark_action.setToolTip("Save current position as bookmark")
        self.add_bookmark_action.triggered.connect(self.add_bookmark)
        toolbar.addAction(self.add_bookmark_action)

        self.goto_bookmark_action = QAction(get_bookmark_icon, "Bookmarks", self)
        self.goto_bookmark_action.setToolTip("View and manage bookmarks")
        self.goto_bookmark_action.triggered.connect(self.goto_bookmark)
        toolbar.addAction(self.goto_bookmark_action)

        # Add clear cache action
        self.clear_cache_action = QAction("Clear Cache", self)
        self.clear_cache_action.setToolTip("Clear all cached audio files and reset playback")
        self.clear_cache_action.triggered.connect(self.clear_cache)
        toolbar.addAction(self.clear_cache_action)

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
        # Save application state
        self.save_state()

        # Stop any ongoing playback
        if self.is_playing:
            # Stop media player
            self.media_player.stop()

            # Stop TTS engine properly
            self.tts_engine.stop()

            # Update UI state
            self.is_playing = False
            self.highlight_timer.stop()

        # Wait for any existing threads to finish
        if hasattr(self, 'synthesis_thread') and self.synthesis_thread and self.synthesis_thread.is_alive():
            try:
                print("Waiting for synthesis thread to finish...")
                self.synthesis_thread.join(1.0)  # Wait up to 1 second
            except Exception as e:
                print(f"Warning: Could not join synthesis thread: {str(e)}")

        if hasattr(self, 'playback_thread') and self.playback_thread and self.playback_thread.is_alive():
            try:
                print("Waiting for playback thread to finish...")
                self.playback_thread.join(1.0)  # Wait up to 1 second
            except Exception as e:
                print(f"Warning: Could not join playback thread: {str(e)}")

        # Unload the TTS model to free resources
        try:
            print("Unloading TTS model...")
            self.tts_engine.unload_model()
        except Exception as e:
            print(f"Warning: Could not unload TTS model: {str(e)}")

        # Accept the close event
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

        # Prepare speech synthesis but don't auto-play
        if self.auto_play:
            # Only auto-play if explicitly enabled
            self.synthesize_speech()
        else:
            # Just prepare the text without playing
            self.text_edited = True
            self.text_display.setReadOnly(False)
            self.status_bar.showMessage("Transcription complete. Press play to start synthesis.")

    def load_file(self, file_path: str):
        """
        Load a text file.
        The file will be automatically converted to markdown and stored in the markdown directory.
        Any edits will be saved to the markdown version.

        Args:
            file_path: Path to the text file.
        """
        try:
            # Show progress
            self.status_bar.showMessage(f"Loading file: {os.path.basename(file_path)}")
            self.progress_bar.setVisible(True)

            # Load the file - this will automatically use the markdown version if it exists
            # or create a new markdown version if it doesn't
            content, format_str = self.text_processor.load_file(file_path)
            self.status_bar.showMessage(f"Loaded file: {os.path.basename(file_path)}")

            # Update UI
            self.current_file_path = file_path
            self.current_text = content

            # Split text into pages
            self.split_text_into_pages(content)

            # Display the first page
            if self.pages:
                self.text_display.setPlainText(self.pages[0])

            # Make text editable immediately
            self.text_display.setReadOnly(False)
            self.status_bar.showMessage("Text is editable. Edit as needed, then press Play to synthesize.")

            # Prepare speech synthesis but don't auto-play
            if self.auto_play:
                # Only auto-play if explicitly enabled
                self.synthesize_speech()
            else:
                # Just prepare the text without playing
                self.text_edited = True
                print("File loaded. Press play to start synthesis.")

            self.status_bar.showMessage(f"Loaded file: {os.path.basename(file_path)}")
            self.progress_bar.setVisible(False)

        except Exception as e:
            self.progress_bar.setVisible(False)
            self.status_bar.showMessage("")
            QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")

    def synthesize_speech(self, start_position=None):
        """
        Synthesize speech from the current page.

        Args:
            start_position: Optional position in the text to start synthesis from.
        """
        if not self.pages or self.current_page_index >= len(self.pages):
            return

        # Get the current page text
        current_page_text = self.text_display.toPlainText()

        # Save any edits to the current page
        if not self.text_display.isReadOnly():
            self.pages[self.current_page_index] = current_page_text

            # Save the edited text if we have a current file path
            if self.current_file_path:
                try:
                    # Rebuild the full text
                    self.current_text = '\n\n'.join(self.pages)

                    # Save directly to the markdown file
                    markdown_path = self.text_processor.save_markdown(self.current_text, self.current_file_path)
                    print(f"Saved edited content to {markdown_path} when synthesizing speech")
                except Exception as e:
                    print(f"Error saving edited file when synthesizing speech: {str(e)}")

        # If the page is empty, don't synthesize
        if not current_page_text.strip():
            self.status_bar.showMessage("No text to synthesize on this page.")
            return

        # Check if this page has been preprocessed
        task_id = f"page_{self.current_page_index}"
        preprocessed_result = self.background_processor.get_result(task_id)

        if preprocessed_result:
            print(f"Using preprocessed content for synthesis of page {self.current_page_index}")
            # We have preprocessed content, use it
            self.current_audio_path = preprocessed_result.get("audio_path")
            self.word_timings = preprocessed_result.get("word_timings")

            # If we have a valid audio path, we can skip synthesis and just play
            if self.current_audio_path and os.path.exists(self.current_audio_path):
                print(f"Using cached audio: {self.current_audio_path}")
                # Convert the file path to a QUrl
                file_url = QUrl.fromLocalFile(self.current_audio_path)
                self.media_player.setSource(file_url)
                self.media_player.play()

                # Update UI
                self.play_button.setIcon(self.pause_icon)
                self.text_display.setReadOnly(True)
                self.highlight_timer.start(250)  # Slower highlighting
                self.is_playing = True
                self.status_bar.showMessage(f"Playing page {self.current_page_index + 1} (using cached audio)")
                return

        # Get TTS settings
        tts_settings = self.state_manager.get("tts_settings", {})
        voice = tts_settings.get("voice", "af_sarah")
        speed = tts_settings.get("speed", 1.0)

        # Stop any existing playback
        if self.is_playing:
            # Stop media player
            self.media_player.stop()

            # Stop TTS engine properly
            self.tts_engine.stop()

            # Update UI state
            self.is_playing = False
            self.play_button.setIcon(self.play_icon)
            self.highlight_timer.stop()

            # Clear any highlighting
            clear_cursor = QTextCursor(self.text_display.document())
            clear_cursor.select(QTextCursor.SelectionType.Document)
            clear_cursor.setCharFormat(QTextCharFormat())

        # Wait for any existing threads to finish
        if hasattr(self, 'synthesis_thread') and self.synthesis_thread and self.synthesis_thread.is_alive():
            try:
                self.synthesis_thread.join(1.0)  # Wait up to 1 second
            except Exception as e:
                print(f"Warning: Could not join synthesis thread: {str(e)}")

        # Clear any existing audio path
        self.current_audio_path = None

        # Reset UI state
        self.play_button.setIcon(self.pause_icon)
        self.text_display.setReadOnly(True)
        self.highlight_timer.stop()

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage(f"Synthesizing page {self.current_page_index + 1}... (will start playing automatically)")

        # Define callback for chunk progress
        def chunk_callback(chunk_index, total_chunks, audio_path, word_timings):
            # Update progress
            progress = int((chunk_index + 1) / total_chunks * 100)
            self.progress_bar.setValue(progress)

            # If this is the first chunk, start playback
            if chunk_index == 0:
                self.handle_first_chunk(audio_path, word_timings)

                # If we have a start position, set it
                if start_position is not None and word_timings:
                    # Find the closest word timing to the start position
                    try:
                        closest_pos = min(word_timings.keys(), key=lambda x: abs(x - start_position))
                        time_sec = word_timings[closest_pos]
                        print(f"Setting initial playback position to {time_sec}s based on cursor at position {start_position}")

                        # Set the position in the TTS engine
                        self.tts_engine.set_position(time_sec)
                    except Exception as e:
                        print(f"Error setting initial position: {str(e)}")

        try:
            # Clear the cache first to avoid any conflicts
            self.tts_engine.clear_all_cache()

            # Start progressive synthesis and playback for the current page only
            self.synthesis_thread, self.playback_thread = self.tts_engine.synthesize_and_play_progressively(
                current_page_text,
                voice=voice,
                speed=speed,
                callback=chunk_callback
            )

            # Update UI state
            self.is_playing = True
            self.play_button.setIcon(self.pause_icon)
            self.text_display.setReadOnly(True)
            self.highlight_timer.start(250)  # Update highlight every 250ms (slower)

            # Reset the last highlighted index
            if hasattr(self, 'last_highlighted_index'):
                delattr(self, 'last_highlighted_index')

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

        # Store both the list and dictionary versions of word timings
        # The list version is used for highlighting
        if hasattr(self.tts_engine, 'word_timings_list'):
            print(f"Using word_timings_list from TTS engine with {len(self.tts_engine.word_timings_list)} entries")
        else:
            print("No word_timings_list available from TTS engine")

        # The dictionary version is used for cursor position lookup
        self.word_timings = word_timings
        print(f"Stored word_timings dictionary with {len(self.word_timings) if self.word_timings else 0} entries")

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
                # Get the current page text
                current_page_text = self.text_display.toPlainText()

                # Synthesize using the chunked approach
                audio_path, word_timings = self.tts_engine.synthesize(
                    current_page_text,
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

                # Get the current page text
                current_page_text = self.text_display.toPlainText()

                # Use dummy synthesis
                audio_path, word_timings = self.tts_engine._dummy_synthesize(current_page_text, speed)

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

        # Store both the list and dictionary versions of word timings
        # The list version is used for highlighting
        if hasattr(self.tts_engine, 'word_timings_list'):
            print(f"Using word_timings_list from TTS engine with {len(self.tts_engine.word_timings_list)} entries")
        else:
            print("No word_timings_list available from TTS engine")

        # The dictionary version is used for cursor position lookup
        self.word_timings = word_timings
        print(f"Stored word_timings dictionary with {len(self.word_timings) if self.word_timings else 0} entries")

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
        try:
            if not self.is_playing:
                return

            # Check if we have word timings
            word_timings_list = None

            # First try to get the list version from the TTS engine
            if hasattr(self.tts_engine, 'word_timings_list') and self.tts_engine.word_timings_list:
                word_timings_list = self.tts_engine.word_timings_list
                print(f"Using word_timings_list from TTS engine with {len(word_timings_list)} entries")
            # If not available, check if we have a dictionary version
            elif self.word_timings:
                print("No word_timings_list available, using dictionary version")
                # We can't highlight properly with just the dictionary, so return
                return
            else:
                # No word timings available
                return

            # Get current time - either from media player or from TTS engine
            if hasattr(self, 'synthesis_thread') and self.synthesis_thread and self.synthesis_thread.is_alive():
                # For progressive playback, use the TTS engine's current position
                current_time = self.tts_engine.current_position
            else:
                # For media player playback, convert from milliseconds to seconds
                current_time = self.media_player.position() / 1000

            # Find the current word index based on time
            current_word_index = -1
            for i, timing in enumerate(word_timings_list):
                if timing["start"] <= current_time <= timing["end"]:
                    current_word_index = i
                    break

            # If we didn't find an exact match, find the closest word
            if current_word_index == -1:
                # Find the word that's about to be spoken
                for i, timing in enumerate(word_timings_list):
                    if timing["start"] > current_time:
                        if i > 0:
                            current_word_index = i - 1
                        else:
                            current_word_index = 0
                        break

            # If we still don't have a word, use the last word
            if current_word_index == -1 and word_timings_list:
                current_word_index = len(word_timings_list) - 1

            # If we have a valid word index, highlight it
            if current_word_index >= 0 and current_word_index < len(word_timings_list):
                current_word = word_timings_list[current_word_index]

                # Store the current word index to avoid unnecessary updates
                if hasattr(self, 'last_highlighted_index') and self.last_highlighted_index == current_word_index:
                    return  # Skip if we're already highlighting this word

                self.last_highlighted_index = current_word_index

                # Store the current position in the text for resuming playback
                if current_word_index >= 0 and current_word_index < len(word_timings_list):
                    # Get the character position of this word in the text
                    word_info = word_timings_list[current_word_index]
                    if "position" in word_info:
                        self.last_highlighted_position = word_info["position"]
                        # Also update the last cursor position to match the current highlighted word
                        self.last_cursor_position = self.last_highlighted_position

                # Get the word to highlight
                word = current_word["word"]

                # Save the current cursor position and scroll position
                old_cursor = self.text_display.textCursor()
                old_scroll_value = self.text_display.verticalScrollBar().value()

                # Clear previous highlighting
                clear_cursor = QTextCursor(self.text_display.document())
                clear_cursor.select(QTextCursor.SelectionType.Document)
                clear_cursor.setCharFormat(QTextCharFormat())

                # Create a format for highlighting
                highlight_format = QTextCharFormat()
                highlight_format.setBackground(QColor(255, 255, 0, 200))  # Brighter yellow
                highlight_format.setForeground(QColor(0, 0, 0))  # Black text

                # Calculate approximate position in text
                # This is more efficient than searching through the entire text
                words_before = current_word_index
                approx_char_pos = min(words_before * 6, self.text_display.document().characterCount() - 1)  # Estimate 6 chars per word

                # Start searching from the approximate position
                cursor = QTextCursor(self.text_display.document())
                cursor.setPosition(max(0, approx_char_pos - 100))  # Start a bit before the estimated position

                # Find and highlight the word
                found = False
                search_count = 0
                max_searches = 200  # Limit searches to avoid performance issues

                while not found and search_count < max_searches:
                    search_count += 1

                    # Find the next occurrence of the word
                    search_result = self.text_display.document().find(
                        word,
                        cursor,
                        QTextDocument.FindFlag.FindCaseSensitively
                    )

                    if search_result.isNull():
                        # If not found with exact case, try case-insensitive
                        cursor.setPosition(max(0, approx_char_pos - 100))
                        search_result = self.text_display.document().find(
                            word,
                            cursor,
                            QTextDocument.FindFlag.FindWholeWords
                        )

                    if not search_result.isNull():
                        # Apply highlighting
                        search_result.setCharFormat(highlight_format)

                        # Create a cursor at the highlighted position for scrolling
                        highlight_cursor = QTextCursor(search_result)

                        # Only scroll if we're in playback mode
                        if self.is_playing:
                            # Create a temporary cursor for scrolling only
                            scroll_cursor = QTextCursor(highlight_cursor)

                            # Save the current cursor position
                            original_cursor = self.text_display.textCursor()

                            # Set the temporary cursor to ensure the highlighted word is visible
                            self.text_display.setTextCursor(scroll_cursor)

                            # Scroll to make the highlighted word visible
                            self.text_display.ensureCursorVisible()

                            # Restore the original cursor position
                            self.text_display.setTextCursor(original_cursor)

                        found = True
                        break
                    else:
                        # If not found near the estimated position, try from the beginning
                        if search_count == 1:
                            cursor.setPosition(0)
                        else:
                            # Move forward and try again
                            cursor.movePosition(QTextCursor.MoveOperation.NextWord)

                # If we're not in playback mode, restore the original cursor and scroll position
                if not self.is_playing:
                    self.text_display.setTextCursor(old_cursor)
                    self.text_display.verticalScrollBar().setValue(old_scroll_value)

                if not found:
                    print(f"Warning: Could not find word '{word}' in text to highlight")
        except Exception as e:
            print(f"Error in update_highlight: {str(e)}")
            # Don't let highlighting errors crash the application

    def on_text_changed(self):
        """Handle text changes in the text display."""
        if not self.is_playing and self.current_text:
            # Mark that text has been edited
            self.text_edited = True
            print("Text changed. Marked for re-synthesis.")

            # Start the timer to trigger re-synthesis after a delay
            self.edit_timer.start(2000)  # 2 seconds delay

    def on_content_changed(self):
        """Handle content changes in the virtual text display."""
        # This is called when the visible content changes due to scrolling
        # We can use this to trigger background processing of nearby pages
        self.preprocess_nearby_pages()

    def preprocess_nearby_pages(self):
        """Preprocess nearby pages in the background."""
        if not self.pages:
            return

        # Get the current page index
        current_index = self.current_page_index

        # Define a function to preprocess a page
        def preprocess_page(page_text, voice, speed):
            """Preprocess a page of text."""
            try:
                # Synthesize the page
                audio_path, word_timings = self.tts_engine.synthesize(
                    page_text,
                    voice=voice,
                    speed=speed
                )
                return {
                    "audio_path": audio_path,
                    "word_timings": word_timings
                }
            except Exception as e:
                print(f"Error preprocessing page: {str(e)}")
                return None

        # Get TTS settings
        tts_settings = self.state_manager.get("tts_settings", {})
        voice = tts_settings.get("voice", "af_sarah")
        speed = tts_settings.get("speed", 1.0)

        # Preprocess the next page if available
        if current_index + 1 < len(self.pages) and current_index + 1 not in self.preprocessed_pages:
            next_page = self.pages[current_index + 1]
            task_id = f"page_{current_index + 1}"

            # Add the task to the background processor
            self.background_processor.add_task(
                task_id,
                preprocess_page,
                1,  # priority (higher priority for the next page)
                self.handle_preprocessed_page,  # callback
                next_page,
                voice,
                speed
            )

            # Mark the page as being preprocessed
            self.preprocessed_pages.add(current_index + 1)
            print(f"Preprocessing page {current_index + 1}")

        # Preprocess the previous page if available
        if current_index > 0 and current_index - 1 not in self.preprocessed_pages:
            prev_page = self.pages[current_index - 1]
            task_id = f"page_{current_index - 1}"

            # Add the task to the background processor
            self.background_processor.add_task(
                task_id,
                preprocess_page,
                2,  # priority (lower priority for the previous page)
                self.handle_preprocessed_page,  # callback
                prev_page,
                voice,
                speed
            )

            # Mark the page as being preprocessed
            self.preprocessed_pages.add(current_index - 1)
            print(f"Preprocessing page {current_index - 1}")

    def handle_preprocessed_page(self, task_id, result):
        """Handle a preprocessed page."""
        if result:
            page_index = int(task_id.split('_')[1])
            print(f"Page {page_index} preprocessed successfully")

    def handle_text_edit(self):
        """Handle text edit after the delay timer expires."""
        if self.text_edited:
            # Get the updated text
            new_text = self.text_display.toPlainText()
            print(f"Handling text edit. New text length: {len(new_text)}")

            # Only re-synthesize if the text has actually changed
            if new_text != self.current_text:
                # Update the current page
                self.pages[self.current_page_index] = new_text

                # Rebuild the full text
                self.current_text = '\n\n'.join(self.pages)

                print("Text content changed. Will re-synthesize on play.")

                # Save the edited text if we have a current file path
                if self.current_file_path:
                    try:
                        # Save directly to the markdown file
                        markdown_path = self.text_processor.save_markdown(self.current_text, self.current_file_path)
                        print(f"Saved edited content to {markdown_path}")
                        self.status_bar.showMessage("Text edited and saved. Will re-synthesize on play.")
                    except Exception as e:
                        print(f"Error saving edited file: {str(e)}")
                        self.status_bar.showMessage("Text edited. Will re-synthesize on play.")
                else:
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
            try:
                # Check if we're using progressive playback
                using_progressive = hasattr(self, 'synthesis_thread') and self.synthesis_thread and self.synthesis_thread.is_alive()

                if using_progressive:
                    # Toggle pause in the TTS engine for progressive playback
                    is_paused = self.tts_engine.toggle_pause()
                    print(f"Progressive playback paused: {is_paused}")
                else:
                    # Pause the media player for non-progressive playback
                    self.media_player.pause()
                    print(f"Media player paused")

                # Update UI
                self.play_button.setIcon(self.play_icon)
                self.text_display.setReadOnly(False)
                self.highlight_timer.stop()
                self.is_playing = False
                self.status_bar.showMessage("Paused. Text is now editable.")

                # Store the cursor position for potential rewind
                self.last_cursor_position = self.text_display.textCursor().position()

                # If we have a highlighted position, use that instead of the cursor position
                if hasattr(self, 'last_highlighted_position') and self.last_highlighted_position > 0:
                    self.last_cursor_position = self.last_highlighted_position
                    print(f"Using highlighted position: {self.last_highlighted_position} as cursor position")

                # Store the current playback position
                if using_progressive:
                    # For progressive playback, get position from TTS engine
                    self.last_playback_position = self.tts_engine.current_position
                    print(f"Stored progressive playback position: {self.last_playback_position}s")
                else:
                    # For media player, get position in milliseconds and convert to seconds
                    self.last_playback_position = self.media_player.position() / 1000.0
                    print(f"Stored media player position: {self.last_playback_position}s")
            except Exception as e:
                print(f"Error pausing playback: {str(e)}")
                self.status_bar.showMessage(f"Error pausing: {str(e)}")
                # Try to recover
                self.is_playing = False
                self.play_button.setIcon(self.play_icon)
                self.text_display.setReadOnly(False)
                self.highlight_timer.stop()
        else:
            # Get the current cursor position
            current_cursor_position = self.text_display.textCursor().position()

            # Check if the cursor was moved (rewind requested)
            cursor_moved = hasattr(self, 'last_cursor_position') and current_cursor_position != self.last_cursor_position

            # Check if text was edited
            if self.text_edited:
                print("Text was edited, re-synthesizing")
                self.text_edited = False

                # Stop any existing playback
                self.tts_engine.stop_requested = True
                if hasattr(self, 'synthesis_thread') and self.synthesis_thread:
                    self.synthesis_thread.join(0.5)

                # Clear audio path to force complete re-synthesis
                self.current_audio_path = None

                # Start new synthesis
                self.synthesize_speech()
            elif cursor_moved:
                print(f"Cursor moved from {self.last_cursor_position} to {current_cursor_position}, rewinding")

                # Find the word at the cursor position
                cursor = self.text_display.textCursor()
                cursor.select(QTextCursor.SelectionType.WordUnderCursor)
                word_at_cursor = cursor.selectedText()

                if word_at_cursor:
                    print(f"Word at cursor: '{word_at_cursor}'")

                    # Find the position of this word in the text
                    text_up_to_cursor = self.current_text[:current_cursor_position]
                    word_index = len(text_up_to_cursor.split())

                    # Find the corresponding time in the word timings
                    # Check if we have word_timings_list from the TTS engine
                    if hasattr(self.tts_engine, 'word_timings_list') and self.tts_engine.word_timings_list:
                        if word_index < len(self.tts_engine.word_timings_list):
                            # Use the list version for direct index access
                            target_time = self.tts_engine.word_timings_list[word_index]["start"]
                            print(f"Rewinding to time: {target_time:.2f}s (using word_timings_list at index {word_index})")
                        else:
                            # Try to find the closest word by position
                            print(f"Word index {word_index} out of range, trying to find closest word by position")
                            closest_word = None
                            closest_distance = float('inf')

                            for i, timing in enumerate(self.tts_engine.word_timings_list):
                                if "position" in timing:
                                    distance = abs(timing["position"] - current_cursor_position)
                                    if distance < closest_distance:
                                        closest_distance = distance
                                        closest_word = timing

                            if closest_word:
                                target_time = closest_word["start"]
                                print(f"Found closest word at position {closest_word.get('position')}, time: {target_time:.2f}s")
                            else:
                                # If no word with position found, use the last word
                                if self.tts_engine.word_timings_list:
                                    target_time = self.tts_engine.word_timings_list[-1]["start"]
                                    print(f"Using last word timing, time: {target_time:.2f}s")
                                else:
                                    print("No word timings available, starting from beginning")
                                    target_time = 0.0
                    # Otherwise try to find the closest position in the dictionary
                    elif self.word_timings:
                        try:
                            # Get the position of this word in the text
                            text_up_to_cursor = self.current_text[:current_cursor_position]
                            print(f"Looking for position close to {current_cursor_position} in word_timings with {len(self.word_timings)} entries")

                            # Get all positions as integers, ensuring they're valid
                            try:
                                positions = [int(float(pos)) for pos in self.word_timings.keys() if str(pos).strip()]
                                print(f"Found {len(positions)} valid positions in word_timings")

                                # Print some sample positions for debugging
                                if positions:
                                    sample_positions = positions[:5] if len(positions) > 5 else positions
                                    print(f"Sample positions: {sample_positions}")
                            except Exception as e:
                                print(f"Error converting positions to integers: {str(e)}")
                                positions = []

                            # Find the closest position
                            if positions:
                                closest_pos = min(positions, key=lambda x: abs(x - current_cursor_position))
                                # Convert to string to look up in the dictionary
                                closest_pos_key = str(closest_pos)
                                if closest_pos_key in self.word_timings:
                                    target_time = float(self.word_timings[closest_pos_key])
                                    print(f"Rewinding to time: {target_time:.2f}s (using closest position {closest_pos})")
                                else:
                                    # Try with float key
                                    closest_pos_key = float(closest_pos)
                                    if closest_pos_key in self.word_timings:
                                        target_time = float(self.word_timings[closest_pos_key])
                                        print(f"Rewinding to time: {target_time:.2f}s (using closest position {closest_pos})")
                                    else:
                                        print(f"Closest position {closest_pos} not found in word_timings, starting from beginning")
                                        target_time = 0.0
                            else:
                                print("No positions available in word_timings, starting from beginning")
                                target_time = 0.0
                        except Exception as e:
                            print(f"Error finding time for cursor position: {str(e)}")
                            target_time = 0.0
                    else:
                        print("No word timings available, starting from beginning")
                        target_time = 0.0

                        # For progressive playback
                        if hasattr(self, 'synthesis_thread') and self.synthesis_thread and self.synthesis_thread.is_alive():
                            # Set the current position in the TTS engine
                            self.tts_engine.current_position = target_time

                            # Find the chunk containing this position
                            chunk_index = self.tts_engine.find_chunk_for_position(target_time)
                            if chunk_index >= 0:
                                print(f"Rewinding to chunk {chunk_index}")
                                self.tts_engine.rewind_to_chunk(chunk_index)

                            # Resume playback
                            self.tts_engine.pause_requested = False
                            self.play_button.setIcon(self.pause_icon)
                            self.text_display.setReadOnly(True)
                            self.highlight_timer.start(250)  # Slower highlighting
                            self.is_playing = True
                            self.status_bar.showMessage(f"Rewound to position {format_time(target_time)}")
                        else:
                            # For media player playback
                            position_ms = int(target_time * 1000)
                            self.media_player.setPosition(position_ms)
                            self.media_player.play()
                            self.play_button.setIcon(self.pause_icon)
                            self.text_display.setReadOnly(True)
                            self.highlight_timer.start(250)  # Slower highlighting
                            self.is_playing = True
                            self.status_bar.showMessage(f"Rewound to position {format_time(target_time)}")
                else:
                    print("Could not find word timing for rewind position")
                    # Just start normal playback
                    self._start_or_resume_playback()
            else:
                # No rewind or edit, just start or resume playback
                self._start_or_resume_playback()

    def _start_or_resume_playback(self):
        """Start or resume playback without rewind."""
        print("Starting or resuming playback")
        try:
            # Get the current cursor position
            cursor_pos = self.text_display.textCursor().position()
            print(f"Current cursor position: {cursor_pos}")

            # If we're using progressive playback
            if hasattr(self, 'synthesis_thread') and self.synthesis_thread and self.synthesis_thread.is_alive():
                print("Resuming progressive playback")

                # Check if we have a saved playback position
                if hasattr(self, 'last_playback_position') and self.last_playback_position > 0:
                    # Set the position in the TTS engine
                    print(f"Resuming from saved position: {self.last_playback_position}s")
                    self.tts_engine.set_position(self.last_playback_position)

                    # Find the chunk containing this position
                    chunk_index = self.tts_engine.find_chunk_for_position(self.last_playback_position)
                    if chunk_index >= 0:
                        print(f"Resuming from chunk {chunk_index}")
                        self.tts_engine.rewind_to_chunk(chunk_index)

                # Resume the paused playback
                is_paused = self.tts_engine.toggle_pause()
                print(f"After toggle, is_paused = {is_paused}")

                # If toggle_pause returned True, it means it's now paused (which is not what we want)
                if is_paused:
                    print("Toggle resulted in paused state, toggling again")
                    is_paused = self.tts_engine.toggle_pause()
                    print(f"After second toggle, is_paused = {is_paused}")

                # Update UI
                self.play_button.setIcon(self.pause_icon)
                self.text_display.setReadOnly(True)
                self.highlight_timer.start(250)  # Slower highlighting
                self.is_playing = True
                self.status_bar.showMessage("Resuming playback...")
            else:
                # Start new synthesis if no current playback
                if not self.current_audio_path:
                    print("No audio path, starting new synthesis")
                    # Start synthesis from the current cursor position
                    self.synthesize_speech(start_position=cursor_pos)
                else:
                    print("Using media player for playback")

                    # Check if we have a saved playback position
                    if hasattr(self, 'last_playback_position') and self.last_playback_position > 0:
                        # Use the saved position
                        time_ms = int(self.last_playback_position * 1000)
                        print(f"Resuming from saved position: {self.last_playback_position}s ({time_ms}ms)")
                        self.media_player.setPosition(time_ms)
                    # Otherwise, set the position based on the cursor position if possible
                    elif self.word_timings and cursor_pos > 0:
                        try:
                            # Find the closest word timing to the cursor position
                            # self.word_timings is now a dictionary mapping positions to times
                            print(f"Looking for closest position to cursor position {cursor_pos} in word_timings with {len(self.word_timings)} entries")

                            # Get all positions as integers, ensuring they're valid
                            try:
                                positions = [int(float(pos)) for pos in self.word_timings.keys() if str(pos).strip()]
                                print(f"Found {len(positions)} valid positions in word_timings")

                                # Print some sample positions for debugging
                                if positions:
                                    sample_positions = positions[:5] if len(positions) > 5 else positions
                                    print(f"Sample positions: {sample_positions}")
                            except Exception as e:
                                print(f"Error converting positions to integers: {str(e)}")
                                positions = []

                            # Find the closest position
                            if positions:
                                closest_pos = min(positions, key=lambda x: abs(x - cursor_pos))
                                # Convert to string to look up in the dictionary
                                closest_pos_key = str(closest_pos)
                                if closest_pos_key in self.word_timings:
                                    time_ms = int(float(self.word_timings[closest_pos_key]) * 1000)
                                    print(f"Setting playback position to {time_ms}ms based on cursor at position {cursor_pos} (closest position: {closest_pos})")
                                    self.media_player.setPosition(time_ms)
                                else:
                                    # Try with float key
                                    closest_pos_key = float(closest_pos)
                                    if closest_pos_key in self.word_timings:
                                        time_ms = int(float(self.word_timings[closest_pos_key]) * 1000)
                                        print(f"Setting playback position to {time_ms}ms based on cursor at position {cursor_pos} (closest position: {closest_pos})")
                                        self.media_player.setPosition(time_ms)
                                    else:
                                        print(f"Closest position {closest_pos} not found in word_timings")
                                        # Start from the beginning
                                        self.media_player.setPosition(0)
                            else:
                                print("No valid positions available in word_timings")
                                # Start from the beginning
                                self.media_player.setPosition(0)
                        except Exception as e:
                            print(f"Error finding closest position: {str(e)}")
                            # Start from the beginning if there's an error
                            self.media_player.setPosition(0)

                    # Check if the audio file exists and is valid
                    if self.current_audio_path and os.path.exists(self.current_audio_path):
                        file_size = os.path.getsize(self.current_audio_path)
                        print(f"Audio file exists at {self.current_audio_path}, size: {file_size} bytes")

                        if file_size == 0:
                            print("Warning: Audio file is empty")
                            self.status_bar.showMessage("Error: Audio file is empty")
                            return

                        # Double-check the media source
                        current_source = self.media_player.source().toString()
                        expected_source = QUrl.fromLocalFile(self.current_audio_path).toString()

                        if current_source != expected_source:
                            print(f"Media source mismatch. Current: {current_source}, Expected: {expected_source}")
                            # Reset the source
                            file_url = QUrl.fromLocalFile(self.current_audio_path)
                            self.media_player.setSource(file_url)
                            print(f"Reset media source to: {file_url.toString()}")
                    else:
                        print(f"Audio file does not exist at {self.current_audio_path}")
                        self.status_bar.showMessage("Error: Audio file not found")
                        return

                    # Use the media player for non-progressive playback
                    print("Starting media player playback...")
                    self.media_player.play()

                    # Give it a moment to start
                    time.sleep(0.1)

                    # Check if playback actually started
                    if self.media_player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
                        print(f"Playback failed to start. Media player state: {self.media_player.playbackState()}")
                        print(f"Media player error: {self.media_player.error()}")
                        error_string = self.media_player.errorString()
                        print(f"Error string: {error_string}")

                        # Try to recover
                        print("Attempting to recover by resetting media player...")
                        self.media_player.stop()
                        self.media_player.setSource(QUrl())  # Clear source
                        time.sleep(0.1)

                        # Set source again
                        file_url = QUrl.fromLocalFile(self.current_audio_path)
                        self.media_player.setSource(file_url)
                        time.sleep(0.1)

                        # Try to play again
                        self.media_player.play()
                        time.sleep(0.1)

                        if self.media_player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
                            print("Recovery attempt failed")
                            self.status_bar.showMessage(f"Playback error: {error_string}")
                            return
                        else:
                            print("Recovery successful")

                    # Update UI
                    self.play_button.setIcon(self.pause_icon)
                    self.text_display.setReadOnly(True)
                    self.highlight_timer.start(250)  # Slower highlighting
                    self.is_playing = True
                    print("Playback started successfully")
        except Exception as e:
            print(f"Error in _start_or_resume_playback: {str(e)}")
            self.status_bar.showMessage(f"Error resuming playback: {str(e)}")
            # Try to recover
            if not self.is_playing:
                # Try to start new synthesis
                self.synthesize_speech()

    def add_bookmark(self):
        """Save the current position as a bookmark."""
        if not self.current_text:
            self.status_bar.showMessage("No text loaded. Cannot add bookmark.")
            return

        # Get current position
        if hasattr(self, 'synthesis_thread') and self.synthesis_thread and self.synthesis_thread.is_alive():
            # For progressive playback, use the TTS engine's current position
            current_position = self.tts_engine.current_position
        else:
            # For media player playback, convert from milliseconds to seconds
            current_position = self.media_player.position() / 1000

        # Get current text cursor position
        text_position = self.text_display.textCursor().position()

        # Get current page index
        page_index = self.current_page_index

        # Create bookmark data
        bookmark_data = {
            "position": current_position,
            "text_position": text_position,
            "file_path": self.current_file_path,
            "page_index": page_index,
            "text": self.current_text[:100] + "..." if len(self.current_text) > 100 else self.current_text,
            "timestamp": time.time(),  # Add timestamp for sorting
            "title": f"Bookmark at {format_time(current_position)} - Page {page_index + 1}"
        }

        # Get existing bookmarks
        bookmarks = self.state_manager.get("bookmarks", [])

        # Add new bookmark
        bookmarks.append(bookmark_data)

        # Save to state manager
        self.state_manager.set("bookmarks", bookmarks)

        # Update UI
        self.status_bar.showMessage(f"Bookmark added at position {format_time(current_position)}")

        # Show a confirmation message
        QMessageBox.information(
            self,
            "Bookmark Added",
            f"Bookmark added at position {format_time(current_position)} on page {page_index + 1}.\n\n"
            f"You can access your bookmarks by clicking 'Go to Bookmark' in the toolbar."
        )

    def goto_bookmark(self):
        """Jump to a saved bookmark position."""
        # Get bookmarks from state manager
        bookmarks = self.state_manager.get("bookmarks", [])

        if not bookmarks:
            self.status_bar.showMessage("No bookmarks saved.")
            QMessageBox.information(self, "No Bookmarks", "You haven't saved any bookmarks yet.")
            return

        # Show bookmarks dialog
        dialog = BookmarksDialog(bookmarks, self)
        dialog.bookmark_selected.connect(self.handle_bookmark_selected)
        dialog.bookmarks_modified.connect(lambda bookmarks: self.state_manager.set("bookmarks", bookmarks))
        dialog.exec()

    def handle_bookmark_selected(self, bookmark):
        """
        Handle bookmark selection from the dialog.

        Args:
            bookmark: The selected bookmark data.
        """
        if not bookmark:
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

        # Go to the bookmarked page
        page_index = bookmark.get("page_index", 0)
        if page_index != self.current_page_index:
            # Save any edits to the current page
            if not self.text_display.isReadOnly():
                current_page_text = self.text_display.toPlainText()
                self.pages[self.current_page_index] = current_page_text

                # Save the edited text if we have a current file path
                if self.current_file_path:
                    try:
                        # Rebuild the full text
                        self.current_text = '\n\n'.join(self.pages)

                        # Save directly to the markdown file
                        markdown_path = self.text_processor.save_markdown(self.current_text, self.current_file_path)
                        print(f"Saved edited content to {markdown_path} when jumping to bookmark")
                    except Exception as e:
                        print(f"Error saving edited file when jumping to bookmark: {str(e)}")

            # Set the current page index
            self.current_page_index = page_index

            # Update text display with the page content
            page_text = self.pages[self.current_page_index]

            # Calculate the start position of this page in the full text
            page_start = 0
            for i in range(self.current_page_index):
                page_start += len(self.pages[i])

            # Set the page text with position information
            self.text_display.set_page_text(page_text, page_start)

            # Update page label and navigation buttons
            self.update_page_label()
            self.update_navigation_buttons()

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

    def clear_cache(self):
        """Clear all cached audio files and reset playback state."""
        # Confirm with the user
        response = QMessageBox.question(
            self,
            "Clear Cache",
            "This will stop playback and remove all cached audio files. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if response != QMessageBox.StandardButton.Yes:
            return

        # Stop any current playback
        if self.is_playing:
            self.media_player.stop()
            self.is_playing = False
            self.play_button.setIcon(self.play_icon)
            self.text_display.setReadOnly(False)
            self.highlight_timer.stop()

        # Stop any TTS processes
        self.tts_engine.stop_requested = True
        if hasattr(self, 'synthesis_thread') and self.synthesis_thread:
            self.synthesis_thread.join(0.5)

        # Clear audio path
        self.current_audio_path = None

        # Clear the TTS engine's cache
        self.tts_engine.clear_all_cache()

        # Update UI
        self.status_bar.showMessage("Cache cleared. All audio files removed.")

    def split_text_into_pages(self, text):
        """
        Split text into pages of manageable size.

        Args:
            text: The text to split.

        Returns:
            List of text pages.
        """
        # Store the full text
        self.full_text = text

        # Store the full text in the virtual text display
        self.text_display.full_text = text

        # Split into pages
        self.pages = []

        # Try to split at paragraph boundaries
        paragraphs = text.split('\n\n')
        current_page = ""

        for paragraph in paragraphs:
            # If adding this paragraph would exceed the page size, start a new page
            if len(current_page) + len(paragraph) + 2 > self.page_size and current_page:
                self.pages.append(current_page)
                current_page = paragraph
            else:
                if current_page:
                    current_page += '\n\n' + paragraph
                else:
                    current_page = paragraph

        # Add the last page if it's not empty
        if current_page:
            self.pages.append(current_page)

        # If no pages were created, create at least one
        if not self.pages:
            self.pages = [text]

        # Reset current page index
        self.current_page_index = 0

        # Update page label
        self.update_page_label()

        # Update navigation buttons
        self.update_navigation_buttons()

        # Clear the preprocessed pages set
        self.preprocessed_pages.clear()

        # Start preprocessing nearby pages
        self.preprocess_nearby_pages()

        return self.pages

    def update_page_label(self):
        """Update the page label with current page information."""
        total_pages = len(self.pages)
        self.page_label.setText(f"Page {self.current_page_index + 1} of {total_pages}")

    def update_navigation_buttons(self):
        """Update the state of navigation buttons."""
        # Enable/disable previous page button
        self.prev_page_button.setEnabled(self.current_page_index > 0)

        # Enable/disable next page button
        self.next_page_button.setEnabled(self.current_page_index < len(self.pages) - 1)

    def reset_playback_position(self):
        """Reset the saved playback position."""
        self.last_playback_position = 0
        print("Playback position reset")

    def go_to_previous_page(self):
        """Go to the previous page."""
        if self.current_page_index > 0:
            # Stop any current playback
            if self.is_playing:
                self.toggle_playback()

            # Reset the playback position when changing pages
            self.reset_playback_position()

            # Save any edits to the current page
            if not self.text_display.isReadOnly():
                current_page_text = self.text_display.toPlainText()
                self.pages[self.current_page_index] = current_page_text

                # Save the edited text if we have a current file path
                if self.current_file_path:
                    try:
                        # Rebuild the full text
                        self.current_text = '\n\n'.join(self.pages)

                        # Save directly to the markdown file
                        markdown_path = self.text_processor.save_markdown(self.current_text, self.current_file_path)
                        print(f"Saved edited content to {markdown_path} when changing page")
                    except Exception as e:
                        print(f"Error saving edited file when changing page: {str(e)}")

            # Go to previous page
            self.current_page_index -= 1

            # Check if this page has been preprocessed
            task_id = f"page_{self.current_page_index}"
            preprocessed_result = self.background_processor.get_result(task_id)

            if preprocessed_result:
                print(f"Using preprocessed content for page {self.current_page_index}")
                # We have preprocessed content, use it
                self.current_audio_path = preprocessed_result.get("audio_path")
                self.word_timings = preprocessed_result.get("word_timings")

            # Update text display with the page content
            page_text = self.pages[self.current_page_index]

            # Calculate the start position of this page in the full text
            page_start = 0
            for i in range(self.current_page_index):
                page_start += len(self.pages[i])

            # Set the page text with position information
            self.text_display.set_page_text(page_text, page_start)

            # Update page label and navigation buttons
            self.update_page_label()
            self.update_navigation_buttons()

            # Trigger preprocessing of nearby pages
            self.preprocess_nearby_pages()

            # Update status
            self.status_bar.showMessage(f"Showing page {self.current_page_index + 1} of {len(self.pages)}")

    def go_to_next_page(self):
        """Go to the next page."""
        if self.current_page_index < len(self.pages) - 1:
            # Stop any current playback
            if self.is_playing:
                self.toggle_playback()

            # Reset the playback position when changing pages
            self.reset_playback_position()

            # Save any edits to the current page
            if not self.text_display.isReadOnly():
                current_page_text = self.text_display.toPlainText()
                self.pages[self.current_page_index] = current_page_text

                # Save the edited text if we have a current file path
                if self.current_file_path:
                    try:
                        # Rebuild the full text
                        self.current_text = '\n\n'.join(self.pages)

                        # Save directly to the markdown file
                        markdown_path = self.text_processor.save_markdown(self.current_text, self.current_file_path)
                        print(f"Saved edited content to {markdown_path} when changing page")
                    except Exception as e:
                        print(f"Error saving edited file when changing page: {str(e)}")

            # Go to next page
            self.current_page_index += 1

            # Check if this page has been preprocessed
            task_id = f"page_{self.current_page_index}"
            preprocessed_result = self.background_processor.get_result(task_id)

            if preprocessed_result:
                print(f"Using preprocessed content for page {self.current_page_index}")
                # We have preprocessed content, use it
                self.current_audio_path = preprocessed_result.get("audio_path")
                self.word_timings = preprocessed_result.get("word_timings")

            # Update text display with the page content
            page_text = self.pages[self.current_page_index]

            # Calculate the start position of this page in the full text
            page_start = 0
            for i in range(self.current_page_index):
                page_start += len(self.pages[i])

            # Set the page text with position information
            self.text_display.set_page_text(page_text, page_start)

            # Update page label and navigation buttons
            self.update_page_label()
            self.update_navigation_buttons()

            # Trigger preprocessing of nearby pages
            self.preprocess_nearby_pages()

            # Update status
            self.status_bar.showMessage(f"Showing page {self.current_page_index + 1} of {len(self.pages)}")

    def show_settings(self):
        """Show the settings dialog."""
        dialog = SettingsDialog(self.state_manager, self)
        if dialog.exec():
            # Settings were changed, update as needed
            self.synthesize_speech()
