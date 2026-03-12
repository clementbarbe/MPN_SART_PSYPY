from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QLabel, QSpinBox, QPushButton)

class TemporalJudgementTab(QWidget):
    def __init__(self, parent_menu):
        super().__init__()
        self.parent_menu = parent_menu
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # TRAINING
        run_training_group = QGroupBox("Training")
        run_training_layout = QVBoxLayout()
        training_trials_layout = QHBoxLayout()
        training_trials_layout.addWidget(QLabel("Essais Training :"))
        self.spin_training_trials = QSpinBox()
        self.spin_training_trials.setRange(1, 200)
        self.spin_training_trials.setValue(12)
        training_trials_layout.addWidget(self.spin_training_trials)
        training_trials_layout.addStretch()
        run_training_layout.addLayout(training_trials_layout)

        btn_run_training = QPushButton("Lancer Training")
        btn_run_training.clicked.connect(self.run_training)
        run_training_layout.addWidget(btn_run_training)
        run_training_group.setLayout(run_training_layout)
        layout.addWidget(run_training_group)

        # BASE
        run_base_group = QGroupBox("Run Base")
        run_base_layout = QVBoxLayout()
        trials_base_layout = QHBoxLayout()
        trials_base_layout.addWidget(QLabel("Essais Base :"))
        self.spin_base_trials = QSpinBox()
        self.spin_base_trials.setRange(1, 120)
        self.spin_base_trials.setValue(72)
        trials_base_layout.addWidget(self.spin_base_trials)
        trials_base_layout.addStretch()
        run_base_layout.addLayout(trials_base_layout)

        trials_block_layout = QHBoxLayout()
        trials_block_layout.addWidget(QLabel("Essais Block :"))
        self.spin_base_block = QSpinBox()
        self.spin_base_block.setRange(1, 120)
        self.spin_base_block.setValue(24)
        trials_block_layout.addWidget(self.spin_base_block)
        trials_block_layout.addStretch()
        run_base_layout.addLayout(trials_block_layout)

        btn_run_base = QPushButton("Lancer Run Base")
        btn_run_base.clicked.connect(self.run_base)
        run_base_layout.addWidget(btn_run_base)
        run_base_group.setLayout(run_base_layout)
        layout.addWidget(run_base_group)

        # CUSTOM
        run_custom_group = QGroupBox("Run Personnalisé")
        run_custom_layout = QVBoxLayout()
        trials_layout = QHBoxLayout()
        trials_layout.addWidget(QLabel("Essais :"))
        self.spin_custom_trials = QSpinBox()
        self.spin_custom_trials.setRange(1, 500)
        self.spin_custom_trials.setValue(24)
        trials_layout.addWidget(self.spin_custom_trials)
        trials_layout.addStretch()
        run_custom_layout.addLayout(trials_layout)

        btn_run_custom = QPushButton("Lancer Custom")
        btn_run_custom.clicked.connect(self.run_custom)
        run_custom_layout.addWidget(btn_run_custom)
        run_custom_group.setLayout(run_custom_layout)
        layout.addWidget(run_custom_group)
        layout.addStretch()

    def get_common(self):
        return {
            'tache': 'TemporalJudgement',
            'isi': (1500, 2500),
            'delays_ms': [200, 300, 400, 550, 700, 800],
            'response_options': [200, 300, 400, 550, 700, 800, 1000, 1200],
            'n_trials_base': 72,      
            'n_trials_block': 24,    
            'n_trials_training': 12   
        }

    def run_training(self):
        params = self.get_common()
        params.update({
            'run_type': 'training',
            'run_id': '00',
            'n_trials_training': self.spin_training_trials.value(),
        })
        self.parent_menu.run_experiment(params)

    def run_base(self):
        params = self.get_common()
        params.update({
            'run_type': 'base',
            'run_id': '00',
            'n_trials_base': self.spin_base_trials.value(),
            'n_trials_block': self.spin_base_block.value(),
        })
        self.parent_menu.run_experiment(params)

    def run_custom(self):
        params = self.get_common()
        params.update({
            'run_type': 'custom',
            'run_id': str(self.spin_custom_idx.value()).zfill(2),
            'n_trials_block': self.spin_custom_trials.value(),
        })
        self.parent_menu.run_experiment(params)