"""
sart.py — Sustained Attention to Response Task (McGill Protocol)
================================================================
Go/NoGo task measuring sustained attention and response inhibition.

Modes (via config):
    • 'training'  → 18 essais randomisés avec feedback
    • 'classic'   → 225 essais (200 Go + 25 NoGo) chargés depuis
                     SART_trials_McGill.xlsx, séquence fixe

Sortie : format identique à McGill_SART_Raw_Data.xlsx
"""

import random
import gc
import os
import math
import numpy as np
from scipy.stats import norm
from psychopy import visual, core, event
import pandas as pd
from datetime import datetime


# =========================================================================
# CONFIG
# =========================================================================
DEFAULT_CONFIG = {
    'mode':               'classic',        # 'training' | 'classic'
    'participant_id':     'P001',
    'session':            '01',
    'target_digit':       3,                 # NoGo digit
    'response_key':       'space',
    'n_trials_training':  18,
    'trial_file':         'SART_trials_McGill.xlsx',
    'isi_range':          (0.300, 0.700),
    'data_dir':           'data/sart',
    'feedback_duration':  0.800,
}


class sart:
    """
    Sustained Attention to Response Task — McGill Protocol.
    """

    # =====================================================================
    # CONSTANTS
    # =====================================================================
    DIGITS              = list(range(1, 10))
    DIGIT_DURATION_S    = 0.250
    MASK_DURATION_S     = 0.900
    MASK_RADIUS         = 0.08
    # Tailles McGill (5 niveaux) — mappées depuis la colonne font_height
    FONT_HEIGHTS        = [0.06, 0.08, 0.10, 0.12, 0.14]

    # =====================================================================
    # __init__
    # =====================================================================
    def __init__(self, win, config=None):
        """
        Args:
            win:    PsychoPy Window
            config: dict de configuration (fusionne avec DEFAULT_CONFIG)
        """
        cfg = {**DEFAULT_CONFIG, **(config or {})}

        self.win              = win
        self.participant_id   = str(cfg['participant_id'])
        self.session          = str(cfg['session'])
        self.mode             = cfg['mode'].lower()
        self.target_digit     = int(cfg['target_digit'])
        self.response_key     = cfg['response_key']
        self.n_trials_train   = int(cfg['n_trials_training'])
        self.trial_file       = cfg['trial_file']
        self.isi_range        = (float(cfg['isi_range'][0]),
                                 float(cfg['isi_range'][1]))
        self.data_dir         = cfg['data_dir']
        self.feedback_dur     = float(cfg['feedback_duration'])

        # State
        self.trial_data  = []
        self.task_clock  = core.Clock()

        # Performance counters
        self.perf = self._empty_perf()

        # Init
        self._measure_frame_rate()
        self._setup_stimuli()
        os.makedirs(self.data_dir, exist_ok=True)

        print(
            f"[SART] Init | mode={self.mode} | target={self.target_digit} | "
            f"fps={self.frame_rate:.1f}"
        )

    # =====================================================================
    # HELPERS
    # =====================================================================

    @staticmethod
    def _empty_perf():
        return {
            'go_correct': 0, 'go_omission': 0,
            'nogo_correct': 0, 'nogo_commission': 0,
            'go_rts': [],
        }

    def _measure_frame_rate(self):
        measured = self.win.getActualFrameRate(
            nIdentical=10, nMaxFrames=100, threshold=1
        )
        self.frame_rate  = measured if measured else 60.0
        self.frame_dur_s = 1.0 / self.frame_rate

        self.digit_n_frames = max(1, round(
            self.DIGIT_DURATION_S / self.frame_dur_s))
        self.mask_n_frames = max(1, round(
            self.MASK_DURATION_S / self.frame_dur_s))

        print(
            f"[SART] Digit={self.digit_n_frames}f "
            f"({self.digit_n_frames * self.frame_dur_s * 1000:.1f}ms) | "
            f"Mask={self.mask_n_frames}f "
            f"({self.mask_n_frames * self.frame_dur_s * 1000:.1f}ms)"
        )

    def _setup_stimuli(self):
        self.digit_stim = visual.TextStim(
            self.win, text='', color='white',
            font='Arial', bold=True, pos=(0.0, 0.0)
        )
        px = 2.0 if self.win.size[1] > 1200 else 1.0
        lw = 3.0 * px

        self.mask_circle = visual.Circle(
            self.win, radius=self.MASK_RADIUS, edges=64,
            lineColor='white', lineWidth=lw, fillColor=None
        )
        arm = self.MASK_RADIUS * 0.80
        self.mask_cross_a = visual.Line(
            self.win, start=(-arm, -arm), end=(arm, arm),
            lineColor='white', lineWidth=lw
        )
        self.mask_cross_b = visual.Line(
            self.win, start=(-arm, arm), end=(arm, -arm),
            lineColor='white', lineWidth=lw
        )
        self.fixation = visual.TextStim(
            self.win, text='+', color='white', height=0.06
        )
        self.fb_symbol = visual.TextStim(
            self.win, text='', height=0.14, pos=(0.0, 0.05)
        )
        self.fb_msg = visual.TextStim(
            self.win, text='', color='white', height=0.045,
            pos=(0.0, -0.15)
        )

    def _draw_digit(self, digit, font_size):
        self.digit_stim.text   = str(digit)
        self.digit_stim.height = font_size
        self.digit_stim.draw()

    def _draw_mask(self):
        self.mask_circle.draw()
        self.mask_cross_a.draw()
        self.mask_cross_b.draw()

    # =====================================================================
    # TRIAL LOADING
    # =====================================================================

    def load_trials_from_excel(self):
        """
        Charge la séquence depuis SART_trials_McGill.xlsx

        Colonnes attendues :
            trialnum | number | dur | font_height | target
               1         2     0.25     0.12         Go
               2         5     0.25     0.08         NoGo
               ...

        Returns:
            list[dict] — un dict par essai
        """
        filepath = self.trial_file
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"[SART] Fichier trial introuvable : {filepath}"
            )

        df = pd.read_excel(filepath)

        # Normaliser noms de colonnes
        df.columns = [c.strip().lower() for c in df.columns]

        required = ['trialnum', 'number', 'font_height', 'target']
        for col in required:
            if col not in df.columns:
                raise KeyError(
                    f"[SART] Colonne '{col}' absente de {filepath}. "
                    f"Colonnes trouvées : {list(df.columns)}"
                )

        trials = []
        for _, row in df.iterrows():
            digit      = int(row['number'])
            font_size  = float(row['font_height'])
            target_str = str(row['target']).strip()

            # Déterminer condition
            if target_str.lower() == 'nogo':
                condition = 'nogo'
            else:
                condition = 'go'

            # Réponse correcte attendue (pour corr)
            # Go → 1 (doit appuyer), NoGo → 0 (ne doit PAS appuyer)
            correct_response = 1 if condition == 'go' else 0

            trials.append({
                'trialnum':         int(row['trialnum']),
                'digit':            digit,
                'font_size':        font_size,
                'condition':        condition,
                'correct_response': correct_response,
                'dur':              float(row.get('dur', self.DIGIT_DURATION_S)),
            })

        print(
            f"[SART] Chargé {len(trials)} essais depuis {filepath} "
            f"({sum(1 for t in trials if t['condition']=='nogo')} NoGo, "
            f"{sum(1 for t in trials if t['condition']=='go')} Go)"
        )
        return trials

    def build_training_trials(self):
        """Génère une séquence training courte avec ≥2 NoGo."""
        n = self.n_trials_train
        reps = max(2, math.ceil(n / len(self.DIGITS)))
        pool = self.DIGITS * reps
        random.shuffle(pool)
        seq = pool[:n]

        nogo_count = seq.count(self.target_digit)
        if nogo_count < 2:
            indices = [i for i, d in enumerate(seq) if d != self.target_digit]
            for idx in random.sample(indices, 2 - nogo_count):
                seq[idx] = self.target_digit
        random.shuffle(seq)

        trials = []
        for i, digit in enumerate(seq, start=1):
            condition = 'nogo' if digit == self.target_digit else 'go'
            trials.append({
                'trialnum':         i,
                'digit':            digit,
                'font_size':        random.choice(self.FONT_HEIGHTS),
                'condition':        condition,
                'correct_response': 1 if condition == 'go' else 0,
                'dur':              self.DIGIT_DURATION_S,
            })
        return trials

    # =====================================================================
    # CORE TRIAL
    # =====================================================================

    def run_trial(self, trial_index, total_trials, trial_info,
                  trials_thisIndex, feedback=False):
        """
        Exécute un essai SART avec timing frame-accurate.

        Returns:
            dict : enregistrement au format McGill
        """
        if event.getKeys(keyList=['escape']):
            raise KeyboardInterrupt("Escape pressed")

        gc.disable()

        digit     = trial_info['digit']
        font_size = trial_info['font_size']
        condition = trial_info['condition']
        is_nogo   = (condition == 'nogo')
        correct_response = trial_info['correct_response']

        responded    = False
        response_key = None
        response_rt  = None

        event.clearEvents('keyboard')

        # =================================================================
        # PHASE 1 — DIGIT
        # =================================================================
        self._draw_digit(digit, font_size)
        self.win.flip()
        digit_onset = self.task_clock.getTime()

        for _ in range(1, self.digit_n_frames):
            if not responded:
                keys = event.getKeys(
                    keyList=[self.response_key],
                    timeStamped=self.task_clock
                )
                if keys:
                    responded    = True
                    response_key = keys[0][0]
                    response_rt  = keys[0][1] - digit_onset
            self._draw_digit(digit, font_size)
            self.win.flip()

        # =================================================================
        # PHASE 2 — MASK
        # =================================================================
        self._draw_mask()
        self.win.flip()

        for _ in range(1, self.mask_n_frames):
            if not responded:
                keys = event.getKeys(
                    keyList=[self.response_key],
                    timeStamped=self.task_clock
                )
                if keys:
                    responded    = True
                    response_key = keys[0][0]
                    response_rt  = keys[0][1] - digit_onset
            self._draw_mask()
            self.win.flip()

        # =================================================================
        # CLASSIFICATION
        # =================================================================
        if is_nogo:
            if responded:
                accuracy = 'commission_error'
                corr     = 0
                self.perf['nogo_commission'] += 1
            else:
                accuracy = 'correct_withhold'
                corr     = 1
                self.perf['nogo_correct'] += 1
        else:
            if responded:
                accuracy = 'correct_go'
                corr     = 1
                self.perf['go_correct'] += 1
                self.perf['go_rts'].append(response_rt)
            else:
                accuracy = 'omission_error'
                corr     = 0
                self.perf['go_omission'] += 1

        # =================================================================
        # FEEDBACK (training)
        # =================================================================
        if feedback:
            is_correct = (corr == 1)
            if is_correct:
                self.fb_symbol.text  = '✓'
                self.fb_symbol.color = 'green'
                if accuracy == 'correct_go':
                    self.fb_msg.text = (
                        f"Correct ! RT : {response_rt*1000:.0f} ms"
                    )
                else:
                    self.fb_msg.text = "Correct ! Bonne inhibition ✓"
            else:
                self.fb_symbol.text  = '✗'
                self.fb_symbol.color = 'red'
                if accuracy == 'commission_error':
                    self.fb_msg.text = (
                        f"Erreur — ne PAS appuyer pour le "
                        f"{self.target_digit}"
                    )
                else:
                    self.fb_msg.text = (
                        "Erreur — appuyez pour les autres chiffres"
                    )

            self.fb_symbol.draw()
            self.fb_msg.draw()
            self.win.flip()
            core.wait(self.feedback_dur)

        gc.enable()
        gc.collect()

        # =================================================================
        # ENREGISTREMENT — Format McGill
        # =================================================================
        # key_resp_2.keys : nom de la touche ou None
        if responded:
            key_val = response_key
        else:
            key_val = None

        # key_resp_2.rt : RT en secondes ou None
        rt_val = round(response_rt, 5) if response_rt is not None else None

        record = {
            'SubjectID':          self.participant_id,
            'session':            self.session,
            'trial_type':         condition,
            'number':             digit,
            'font_height':        font_size,
            'correct':            correct_response,
            'key_resp_2.keys':    key_val,
            'key_resp_2.corr':    corr,
            'key_resp_2.rt':      rt_val,
            'trials.thisN':       trial_index - 1,   # 0-indexed
            'trials.thisIndex':   trials_thisIndex,
            'accuracy':           accuracy,
        }
        self.trial_data.append(record)

        # Console
        rt_str = f"{response_rt*1000:.1f}ms" if response_rt else "  ---  "
        tag    = "NOGO" if is_nogo else " GO "
        print(
            f"  {trial_index:>3}/{total_trials:<3} | "
            f"d={digit} {tag} | "
            f"Resp={'Y' if responded else 'N'} | "
            f"RT={rt_str:>8} | {accuracy}"
        )

        # =================================================================
        # ISI
        # =================================================================
        isi = random.uniform(*self.isi_range)
        self.fixation.draw()
        self.win.flip()
        core.wait(isi)

        return record

    # =====================================================================
    # BLOCK
    # =====================================================================

    def run_block(self, trials, block_name, feedback=False):
        total = len(trials)
        print(f"\n{'='*55}")
        print(
            f"  Block : {block_name} | {total} essais | "
            f"feedback={'ON' if feedback else 'OFF'}"
        )
        print(f"{'='*55}")

        for i, trial_info in enumerate(trials, start=1):
            self.run_trial(
                trial_index=i,
                total_trials=total,
                trial_info=trial_info,
                trials_thisIndex=trial_info['trialnum'] - 1,
                feedback=feedback
            )

        self._print_performance(block_name)

    # =====================================================================
    # METRICS
    # =====================================================================

    def compute_metrics(self):
        p = self.perf
        total_go   = p['go_correct']   + p['go_omission']
        total_nogo = p['nogo_correct'] + p['nogo_commission']

        go_acc   = (p['go_correct']  / total_go   * 100) if total_go   else 0
        nogo_acc = (p['nogo_correct'] / total_nogo * 100) if total_nogo else 0

        if p['go_rts']:
            rts       = np.array(p['go_rts']) * 1000
            mean_rt   = float(np.mean(rts))
            sd_rt     = float(np.std(rts, ddof=1)) if len(rts) > 1 else 0.0
            rtcv      = sd_rt / mean_rt if mean_rt > 0 else 0.0
            median_rt = float(np.median(rts))
        else:
            mean_rt = sd_rt = rtcv = median_rt = 0.0

        hit_rate = (p['nogo_correct'] / total_nogo) if total_nogo else 0.5
        fa_rate  = (p['go_omission']  / total_go)   if total_go   else 0.5
        hit_rate = np.clip(hit_rate, 0.01, 0.99)
        fa_rate  = np.clip(fa_rate,  0.01, 0.99)
        try:
            d_prime = float(norm.ppf(hit_rate) - norm.ppf(fa_rate))
        except Exception:
            d_prime = 0.0

        return {
            'total_go':          total_go,
            'total_nogo':        total_nogo,
            'go_correct':        p['go_correct'],
            'go_omission':       p['go_omission'],
            'nogo_correct':      p['nogo_correct'],
            'nogo_commission':   p['nogo_commission'],
            'go_accuracy_pct':   round(go_acc, 1),
            'nogo_accuracy_pct': round(nogo_acc, 1),
            'mean_rt_ms':        round(mean_rt, 1),
            'median_rt_ms':      round(median_rt, 1),
            'sd_rt_ms':          round(sd_rt, 1),
            'rtcv':              round(rtcv, 4),
            'd_prime':           round(d_prime, 3),
        }

    def _print_performance(self, block_name=""):
        m = self.compute_metrics()
        print(
            f"\n{'─'*55}\n"
            f"  SART Performance — {block_name}\n"
            f"{'─'*55}\n"
            f"  Go  Accuracy : {m['go_accuracy_pct']:5.1f}%  "
            f"({m['go_correct']}/{m['total_go']})\n"
            f"  NoGo Accuracy: {m['nogo_accuracy_pct']:5.1f}%  "
            f"({m['nogo_correct']}/{m['total_nogo']})\n"
            f"  Commissions  : {m['nogo_commission']}\n"
            f"  Omissions    : {m['go_omission']}\n"
            f"  Mean RT      : {m['mean_rt_ms']:6.1f} ms\n"
            f"  Median RT    : {m['median_rt_ms']:6.1f} ms\n"
            f"  SD RT        : {m['sd_rt_ms']:6.1f} ms\n"
            f"  RTCV         : {m['rtcv']:.4f}\n"
            f"  d′           : {m['d_prime']:+.3f}\n"
            f"{'─'*55}"
        )

    # =====================================================================
    # SAVE — Format McGill_SART_Raw_Data
    # =====================================================================

    def save_data(self):
        """
        Sauvegarde au format McGill_SART_Raw_Data.xlsx

        Colonnes de sortie :
            SubjectID | session | trial_type | number | font_height |
            correct | key_resp_2.keys | key_resp_2.corr | key_resp_2.rt |
            trials.thisN | trials.thisIndex | accuracy
        """
        if not self.trial_data:
            print("[SART] Aucune donnée à sauvegarder.")
            return None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename  = (
            f"McGill_SART_Raw_Data_"
            f"{self.participant_id}_ses-{self.session}_"
            f"{self.mode}_{timestamp}.xlsx"
        )
        filepath = os.path.join(self.data_dir, filename)

        # Colonnes dans l'ordre McGill
        col_order = [
            'SubjectID',
            'session',
            'trial_type',
            'number',
            'font_height',
            'correct',
            'key_resp_2.keys',
            'key_resp_2.corr',
            'key_resp_2.rt',
            'trials.thisN',
            'trials.thisIndex',
            'accuracy',
        ]

        df = pd.DataFrame(self.trial_data)
        # Réordonner et ne garder que les colonnes attendues
        df = df[[c for c in col_order if c in df.columns]]

        df.to_excel(filepath, index=False, engine='openpyxl')
        print(f"[SART] Données sauvegardées → {filepath}")
        return filepath

    # =====================================================================
    # INSTRUCTIONS
    # =====================================================================

    def show_instructions(self, text):
        instr = visual.TextStim(
            self.win, text=text, color='white',
            height=0.045, wrapWidth=1.4
        )
        instr.draw()
        self.win.flip()
        event.waitKeys()

    # =====================================================================
    # MAIN RUN
    # =====================================================================

    def run(self):
        """
        Point d'entrée principal.
            mode='training' → feedback, séquence aléatoire
            mode='classic'  → pas de feedback, séquence Excel McGill
        """
        filepath = None

        try:
            # ─── Instructions ───────────────────────────────────────
            if self.mode == 'training':
                instr = (
                    "ENTRAÎNEMENT — SART\n\n"
                    "Des chiffres de 1 à 9 vont apparaître.\n\n"
                    f"  ➤ Appuyez sur [{self.response_key.upper()}] "
                    f"pour TOUS les chiffres\n"
                    f"     SAUF le {self.target_digit}.\n\n"
                    f"  ➤ Pour le {self.target_digit} : "
                    f"NE PAS appuyer.\n\n"
                    "Répondez vite et précisément.\n\n"
                    f"{self.n_trials_train} essais (avec feedback).\n\n"
                    "Appuyez pour commencer…"
                )
            else:
                instr = (
                    "SART — Attention Soutenue\n\n"
                    "Des chiffres de 1 à 9 vont apparaître.\n\n"
                    f"  ➤ Appuyez sur [{self.response_key.upper()}] "
                    f"pour TOUS les chiffres\n"
                    f"     SAUF le {self.target_digit}.\n\n"
                    f"  ➤ Pour le {self.target_digit} : "
                    f"NE PAS appuyer.\n\n"
                    "Répondez vite et précisément.\n\n"
                    "225 essais. Bonne concentration !\n\n"
                    "Appuyez pour commencer…"
                )

            self.show_instructions(instr)
            self.task_clock.reset()

            # ─── Bloc ───────────────────────────────────────────────
            if self.mode == 'training':
                trials = self.build_training_trials()
                self.run_block(
                    trials, block_name="TRAINING", feedback=True
                )
            else:
                trials = self.load_trials_from_excel()
                self.run_block(
                    trials, block_name="CLASSIC", feedback=False
                )

            print("\n[SART] Tâche terminée ✓")

        except KeyboardInterrupt:
            print("\n[SART] Interrompu par l'utilisateur.")

        finally:
            filepath = self.save_data()
            self._print_performance("FINAL")

            self.show_instructions(
                "Fin de la tâche.\n\nMerci pour votre participation."
            )

        return filepath


# =========================================================================
# STANDALONE
# =========================================================================
if __name__ == '__main__':
    from psychopy import visual as vis

    win = vis.Window(
        size=(1920, 1080), fullscr=False,
        color='black', units='norm'
    )

    # ── Training ──
    # config = {
    #     'mode':            'training',
    #     'participant_id':  'TEST_001',
    #     'target_digit':    3,
    #     'response_key':    'space',
    # }

    # ── Classic (McGill) ──
    config = {
        'mode':            'classic',
        'participant_id':  'TEST_001',
        'session':         '01',
        'target_digit':    3,
        'response_key':    'space',
        'trial_file':      'SART_trials_McGill.xlsx',
        'data_dir':        'data/sart',
    }

    task = SART(win=win, config=config)
    saved = task.run()

    win.close()
    core.quit()