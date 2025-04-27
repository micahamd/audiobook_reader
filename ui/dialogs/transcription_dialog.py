"""
Transcription dialog module for the Audiobook Reader application.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QDialogButtonBox, QCheckBox
)


class TranscriptionDialog(QDialog):
    """Dialog for configuring audio transcription."""
    
    def __init__(self, parent=None):
        """
        Initialize the dialog.
        
        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        
        self.setWindowTitle("Transcribe Audio")
        self.resize(400, 200)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Add description
        description = QLabel(
            "Would you like to transcribe this audio file to text?\n\n"
            "Transcription will use OpenAI's Whisper model to convert speech to text."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        
        # Add model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.model_combo.setCurrentText("base")
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)
        
        # Add language selection
        language_layout = QHBoxLayout()
        language_label = QLabel("Language:")
        self.language_combo = QComboBox()
        self.language_combo.addItem("Auto-detect", None)
        self.language_combo.addItem("English", "en")
        self.language_combo.addItem("Spanish", "es")
        self.language_combo.addItem("French", "fr")
        self.language_combo.addItem("German", "de")
        self.language_combo.addItem("Italian", "it")
        self.language_combo.addItem("Japanese", "ja")
        self.language_combo.addItem("Chinese", "zh")
        language_layout.addWidget(language_label)
        language_layout.addWidget(self.language_combo)
        layout.addLayout(language_layout)
        
        # Add auto-play checkbox
        self.auto_play = QCheckBox("Auto-play after transcription")
        self.auto_play.setChecked(True)
        layout.addWidget(self.auto_play)
        
        # Add buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def get_model(self) -> str:
        """
        Get the selected model.
        
        Returns:
            The model name.
        """
        return self.model_combo.currentText()
    
    def get_language(self) -> str:
        """
        Get the selected language.
        
        Returns:
            The language code or None for auto-detect.
        """
        return self.language_combo.currentData()
    
    def get_auto_play(self) -> bool:
        """
        Get the auto-play setting.
        
        Returns:
            True if auto-play is enabled, False otherwise.
        """
        return self.auto_play.isChecked()
