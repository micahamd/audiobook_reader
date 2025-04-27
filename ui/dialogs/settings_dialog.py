"""
Settings dialog module for the Audiobook Reader application.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QDialogButtonBox, QSlider, QGroupBox,
    QFormLayout
)
from PyQt6.QtCore import Qt

from core.state_manager import StateManager


class SettingsDialog(QDialog):
    """Dialog for configuring application settings."""

    def __init__(self, state_manager: StateManager, parent=None):
        """
        Initialize the dialog.

        Args:
            state_manager: The state manager.
            parent: Parent widget.
        """
        super().__init__(parent)

        self.state_manager = state_manager

        self.setWindowTitle("Settings")
        self.resize(400, 300)

        self.setup_ui()
        self.load_settings()

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # TTS settings group
        tts_group = QGroupBox("Text-to-Speech Settings")
        tts_layout = QFormLayout()

        # Voice selection
        self.voice_combo = QComboBox()
        self.voice_combo.addItem("Sarah (Female)", "af_sarah")
        self.voice_combo.addItem("Nicole (Female)", "af_nicole")
        self.voice_combo.addItem("Sky (Female)", "af_sky")
        self.voice_combo.addItem("Adam (Male)", "am_adam")
        self.voice_combo.addItem("Michael (Male)", "am_michael")
        self.voice_combo.addItem("Emma (Female)", "bf_emma")
        self.voice_combo.addItem("Isabella (Female)", "bf_isabella")
        self.voice_combo.addItem("George (Male)", "bm_george")
        self.voice_combo.addItem("Lewis (Male)", "bm_lewis")
        tts_layout.addRow("Voice:", self.voice_combo)

        # Speed slider
        speed_layout = QHBoxLayout()
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(50, 200)
        self.speed_slider.setValue(100)
        self.speed_label = QLabel("1.0x")
        self.speed_slider.valueChanged.connect(self.update_speed_label)
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(self.speed_label)
        tts_layout.addRow("Speed:", speed_layout)

        tts_group.setLayout(tts_layout)
        layout.addWidget(tts_group)

        # STT settings group
        stt_group = QGroupBox("Speech-to-Text Settings")
        stt_layout = QFormLayout()

        # Model selection
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        stt_layout.addRow("Model:", self.model_combo)

        # Language selection
        self.language_combo = QComboBox()
        self.language_combo.addItem("Auto-detect", None)
        self.language_combo.addItem("English", "en")
        self.language_combo.addItem("Spanish", "es")
        self.language_combo.addItem("French", "fr")
        self.language_combo.addItem("German", "de")
        self.language_combo.addItem("Italian", "it")
        self.language_combo.addItem("Japanese", "ja")
        self.language_combo.addItem("Chinese", "zh")
        stt_layout.addRow("Language:", self.language_combo)

        stt_group.setLayout(stt_layout)
        layout.addWidget(stt_group)

        # Add buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.save_settings)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def load_settings(self):
        """Load settings from the state manager."""
        # Load TTS settings
        tts_settings = self.state_manager.get("tts_settings", {})
        voice = tts_settings.get("voice", "default")
        speed = tts_settings.get("speed", 1.0)

        # Set voice
        index = self.voice_combo.findData(voice)
        if index >= 0:
            self.voice_combo.setCurrentIndex(index)

        # Set speed
        self.speed_slider.setValue(int(speed * 100))

        # Load STT settings
        stt_settings = self.state_manager.get("stt_settings", {})
        model = stt_settings.get("model", "base")
        language = stt_settings.get("language", None)

        # Set model
        index = self.model_combo.findText(model)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)

        # Set language
        index = self.language_combo.findData(language)
        if index >= 0:
            self.language_combo.setCurrentIndex(index)

    def update_speed_label(self, value):
        """
        Update the speed label.

        Args:
            value: The slider value.
        """
        speed = value / 100.0
        self.speed_label.setText(f"{speed:.1f}x")

    def save_settings(self):
        """Save settings to the state manager."""
        # Save TTS settings
        tts_settings = {
            "voice": self.voice_combo.currentData(),
            "speed": self.speed_slider.value() / 100.0
        }
        self.state_manager.set("tts_settings", tts_settings)

        # Save STT settings
        stt_settings = {
            "model": self.model_combo.currentText(),
            "language": self.language_combo.currentData()
        }
        self.state_manager.set("stt_settings", stt_settings)

        # Save state
        self.state_manager.save_state()

        # Accept dialog
        self.accept()
