# tabs_sart.py

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QLabel, QSpinBox, QPushButton, QComboBox)


class SartTab(QWidget):
    def __init__(self, parent_menu):
        super().__init__()
        self.parent_menu = parent_menu
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # ── TRAINING ────────────────────────────────────────────
        training_group = QGroupBox("Training")
        training_layout = QVBoxLayout()

        btn_training = QPushButton("Lancer Training")
        btn_training.clicked.connect(self.run_training)
        training_layout.addWidget(btn_training)

        training_group.setLayout(training_layout)
        layout.addWidget(training_group)

        # ── TEST ────────────────────────────────────
        classic_group = QGroupBox("TEST")
        classic_layout = QVBoxLayout()

        btn_classic = QPushButton("Lancer Test")
        btn_classic.clicked.connect(self.run_classic)
        classic_layout.addWidget(btn_classic)

        classic_group.setLayout(classic_layout)
        layout.addWidget(classic_group)

        layout.addStretch()

        # ── TRAINING + TEST ────────────────────────────────────
        classic_group = QGroupBox("TRAINING + TEST)")
        classic_layout = QVBoxLayout()

        btn_classic = QPushButton("Lancer Full")
        btn_classic.clicked.connect(self.run_full)
        classic_layout.addWidget(btn_classic)

        classic_group.setLayout(classic_layout)
        layout.addWidget(classic_group)

        layout.addStretch()

    # ── helpers ──────────────────────────────────────────────────

    def get_common(self):
        return {
            'tache':            'sart',
            'target_digit':     3,
            'response_key':     'space',
            'isi_range':        (0.300, 0.700),
            'trial_file':       'SART_trials_McGill.xlsx',
        }

    def run_training(self):
        params = self.get_common()
        params.update({
            'run_type':          'training',
        })
        self.parent_menu.run_experiment(params)

    def run_classic(self):
        params = self.get_common()
        params.update({
            'run_type': 'test',
        })
        self.parent_menu.run_experiment(params)
        
    def run_full(self):
        params = self.get_common()
        params.update({
            'run_type': 'full',
        })
        self.parent_menu.run_experiment(params)