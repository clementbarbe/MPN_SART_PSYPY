from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QTabWidget, QLineEdit, QCheckBox, QLabel,
                            QSpinBox, QGroupBox, QMessageBox)
from PyQt6.QtGui import QFont
import sys

# Direct imports for task tabs
from gui.tabs.tabs_sart import SartTab
from utils.utils import is_valid_name
from utils.logger import get_logger

logger = get_logger()

class ExperimentMenu(QMainWindow):
    def __init__(self, last_config=None):
        super().__init__()
        self.setWindowTitle("Configuration Expérimentale")
        
        # --- POLICE GLOBALE TAILLE 12 ---
        self.global_font = QFont("Segoe UI", 12)
        self.setFont(self.global_font)
        
        # Fenêtre redimensionnée pour le confort visuel
        self.setFixedSize(800, 550)
        
        self.final_config = None

        self.default_config = {
            'nom': '', 'enregistrer': True, 
            'fullscr': True, 'screenid': 1
        }

        if last_config:
            self.default_config.update(last_config)

        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        main_widget.setLayout(main_layout)
        
        self.create_general_section(main_layout)
        self.create_task_tabs(main_layout)

    def create_general_section(self, parent_layout):
        group = QGroupBox("Configuration Générale")
        layout = QHBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(15, 20, 15, 15)

        # -- Champ Participant --
        layout.addWidget(QLabel("ID Participant:"))
        self.txt_name = QLineEdit()
        self.txt_name.setFixedWidth(180)
        self.txt_name.setText(self.default_config.get('nom', ''))
        layout.addWidget(self.txt_name)
        
        # -- Champ Écran --
        layout.addWidget(QLabel("Écran:"))
        self.screenid = QSpinBox()
        self.screenid.setRange(1, len(QApplication.screens()))
        self.screenid.setFixedWidth(75)
        saved_screen = self.default_config.get('screenid', 1)
        self.screenid.setValue(saved_screen + 1)
        layout.addWidget(self.screenid)
        
        # -- Case à cocher Enregistrer --
        self.chk_save = QCheckBox("Enregistrer")
        self.chk_save.setChecked(self.default_config.get('enregistrer', True))
        layout.addWidget(self.chk_save)

        layout.addStretch()
        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_task_tabs(self, parent_layout):
        self.tabs = QTabWidget()
        self.tabs.addTab(SartTab(self), "Temporal Judgement")
        parent_layout.addWidget(self.tabs)

    def validate_config(self):
        nom = self.txt_name.text().strip()
        if not is_valid_name(nom):
            QMessageBox.warning(self, "Erreur", "ID Participant invalide.")
            return None
        
        config = self.default_config.copy()
        config.update({
            'nom': nom,
            'enregistrer': self.chk_save.isChecked(),
            'screenid': self.screenid.value() - 1
        })
        return config

    def run_experiment(self, task_params):
        general_config = self.validate_config()
        if not general_config: return
        self.final_config = {**general_config, **task_params}
        self.close()
        QApplication.instance().quit()

    def get_config(self):
        return self.final_config

    def closeEvent(self, event):
        event.accept()

def show_qt_menu(last_config=None):
    app = QApplication.instance() or QApplication(sys.argv)
    menu = ExperimentMenu(last_config)
    menu.show()
    app.exec()
    return menu.get_config()