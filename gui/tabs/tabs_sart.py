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

        trials_row = QHBoxLayout()
        trials_row.addWidget(QLabel("Essais Training :"))
        self.spin_training_trials = QSpinBox()
        self.spin_training_trials.setRange(1, 100)
        self.spin_training_trials.setValue(18)
        trials_row.addWidget(self.spin_training_trials)
        trials_row.addStretch()
        training_layout.addLayout(trials_row)

        btn_training = QPushButton("Lancer Training")
        btn_training.clicked.connect(self.run_training)
        training_layout.addWidget(btn_training)

        training_group.setLayout(training_layout)
        layout.addWidget(training_group)

        # ── CLASSIC (McGill) ────────────────────────────────────
        classic_group = QGroupBox("Classic (McGill — 225 essais)")
        classic_layout = QVBoxLayout()

        btn_classic = QPushButton("Lancer Classic")
        btn_classic.clicked.connect(self.run_classic)
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
            'n_trials_training': 18,
        }

    def run_training(self):
        params = self.get_common()
        params.update({
            'run_type':          'training',
            'run_id':            '00',
            'n_trials_training': self.spin_training_trials.value(),
        })
        self.parent_menu.run_experiment(params)

    def run_classic(self):
        params = self.get_common()
        params.update({
            'run_type': 'classic',
            'run_id':   '00',
        })
        self.parent_menu.run_experiment(params)