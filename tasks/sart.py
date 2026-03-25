"""
sart.py — Sustained Attention to Response Task (McGill Protocol)
================================================================
Modes disponibles (paramètre 'mode') :
    'full'      → screen1 → screen2 → fixation(2 s) → 18 essais training → screen3 → fixation(2 s) → 225 essais test
    'training'  → screen1 → screen2 → fixation(2 s) → 18 essais training
    'test'      → screen1 → screen2 → fixation(2 s) → 225 essais test

Paramètre 'training_feedback' (bool, défaut True) :
    True  → feedback visuel après chaque essai d'entraînement
    False → entraînement silencieux

Classification latencyType :
    0 → aucune réponse (latency = 1150 ms)
    1 → RT < 100 ms   → Go Anticipatory  / NoGo Failure
    2 → 100 ≤ RT < 200 ms → Go Ambiguous / NoGo Failure
    3 → RT ≥ 200 ms   → Go Success       / NoGo Failure

Entrée  : SART_trials_McGill.xlsx
          · Block='Training' → 18 essais entraînement (16 GO, 2 NO-GO)
          · Block='Main'     → 225 essais test        (200 GO, 25 NO-GO)
Sortie  : McGill_SART_Raw_Data_*.xlsx  (feuilles : Training / Test / All_Trials)
          + data/sart/qc/SART_TimingQC_*.csv
"""

import gc
import os
import numpy as np
import pandas as pd
from datetime import datetime

from psychopy import visual, core
from utils.base_task import BaseTask


# =========================================================================
# CONFIG
# =========================================================================
DEFAULT_CONFIG = {
    'mode':               'full',
    'participant_id':     'P001',
    'session':            '01',
    'target_digit':       3,
    'response_key':       'space',
    'trial_file':         'SART_trials_McGill.xlsx',
    'data_dir':           'data/sart',
    'feedback_duration':  0.800,
    'training_feedback':  True,
}

# 3 modes exacts — aucun alias ambigu
MODE_ALIASES = {
    'full':      'full',           # entraînement + screen3 + test
    'training':  'training_only',  # entraînement uniquement
    'test':      'test_only',      # test uniquement
}

SIZE_MAP = {
    'tiny':       {'pct': '2pct',  'height': 0.04},
    'small':      {'pct': '4pct',  'height': 0.06},
    'medium':     {'pct': '10pct', 'height': 0.10},
    'large':      {'pct': '16pct', 'height': 0.14},
    'very large': {'pct': '16pct', 'height': 0.18},
}

KEY_CODE_MAP   = {'space': 57}
TOTAL_TRIAL_MS = 1150

RT_ANTICIPATORY_MAX = 100   # RT < 100 ms  → type 1
RT_AMBIGUOUS_MAX    = 200   # RT < 200 ms  → type 2 ; sinon type 3

# ── Comptages attendus par bloc (pour validation) ─────────────────────────
TRAINING_N_TRIALS = 18
TRAINING_N_GO     = 16
TRAINING_N_NOGO   = 2
TEST_N_TRIALS     = 225
TEST_N_GO         = 200
TEST_N_NOGO       = 25


class sart(BaseTask):
    """Sustained Attention to Response Task — McGill Protocol."""

    DIGITS              = list(range(1, 10))
    DIGIT_DURATION_S    = 0.250
    MASK_DURATION_S     = 0.900
    MASK_RADIUS         = 0.08
    FIXATION_DURATION_S = 1.0   # durée de la croix de fixation avant chaque bloc

    # =================================================================
    # INIT
    # =================================================================
    def __init__(self, win, config=None):
        cfg = {**DEFAULT_CONFIG, **(config or {})}

        super().__init__(
            win=win,
            nom=cfg['participant_id'],
            session=cfg['session'],
            task_name="SART McGill",
            folder_name=os.path.basename(cfg['data_dir']),
            eyetracker_actif=False,
            parport_actif=False,
            enregistrer=True,
        )

        # ── Normalisation du mode ─────────────────────────────────────────
        raw_mode  = cfg['mode'].lower().strip()
        self.mode = MODE_ALIASES.get(raw_mode)
        if self.mode is None:
            self.logger.warn(
                f"[SART] Mode '{raw_mode}' non reconnu — fallback 'full'. "
                f"Valides : {sorted(MODE_ALIASES.keys())}"
            )
            self.mode = 'full'

        self.target_digit      = int(cfg['target_digit'])
        self.response_key      = cfg['response_key']
        self.trial_file        = cfg['trial_file']
        self.feedback_dur      = float(cfg['feedback_duration'])
        self.training_feedback = bool(cfg.get('training_feedback', True))

        # Données séparées par phase
        self.training_data   = []
        self.training_timing = []
        self.test_data       = []
        self.test_timing     = []

        # Pointeurs actifs (basculés par _set_phase)
        self.trial_data     = self.training_data
        self.timing_log     = self.training_timing
        self.global_records = []

        self.perf = self._empty_perf()

        self._measure_frame_rate()
        self._setup_stimuli()

        self.logger.log(
            f"[SART] Init | mode={self.mode} (raw='{raw_mode}') | "
            f"target={self.target_digit} | "
            f"training_feedback={self.training_feedback} | "
            f"fps={self.frame_rate:.1f} | frame={self.frame_dur_s*1000:.2f}ms"
        )

        self.doqc = False

    # =================================================================
    # HELPERS
    # =================================================================
    @staticmethod
    def _empty_perf():
        return {
            'go_correct': 0, 'go_omission': 0,
            'nogo_correct': 0, 'nogo_commission': 0,
            'go_rts': [],
            'count_anticipatory':     0,
            'correct_suppressions':   0,
            'incorrect_suppressions': 0,
            'count_nogo':             0,
            'count_go':               0,
            'count_valid_go':         0,
        }

    def _set_phase(self, phase):
        """Bascule les pointeurs de données vers la bonne phase."""
        if phase == 'training':
            self.trial_data = self.training_data
            self.timing_log = self.training_timing
        elif phase == 'test':
            self.trial_data = self.test_data
            self.timing_log = self.test_timing
        else:
            raise ValueError(f"[SART] Phase inconnue : {phase}")
        self.global_records = self.trial_data
        self.logger.log(f"[SART] Phase → {phase}")

    @staticmethod
    def _size_text_to_info(size_text):
        key  = str(size_text).strip().lower()
        info = SIZE_MAP.get(key)
        if info:
            return info['pct'], info['height']
        try:
            val = float(size_text)
            return f"{int(val*100)}pct", val
        except (ValueError, TypeError):
            return '10pct', 0.10

    @staticmethod
    def _key_to_code(key_name):
        if key_name is None:
            return 0
        return KEY_CODE_MAP.get(key_name, ord(key_name[0]) if key_name else 0)

    def _measure_frame_rate(self):
        measured = self.win.getActualFrameRate(nIdentical=10, nMaxFrames=100, threshold=1)
        self.frame_rate        = measured if measured else 60.0
        self.frame_dur_s       = 1.0 / self.frame_rate
        self.digit_n_frames    = max(1, round(self.DIGIT_DURATION_S    / self.frame_dur_s))
        self.mask_n_frames     = max(1, round(self.MASK_DURATION_S     / self.frame_dur_s))
        self.fixation_n_frames = max(1, round(self.FIXATION_DURATION_S / self.frame_dur_s))
        self.logger.log(
            f"[SART] Digit={self.digit_n_frames}f "
            f"({self.digit_n_frames * self.frame_dur_s * 1000:.1f}ms) | "
            f"Mask={self.mask_n_frames}f "
            f"({self.mask_n_frames * self.frame_dur_s * 1000:.1f}ms) | "
            f"Fixation={self.fixation_n_frames}f ({self.FIXATION_DURATION_S*1000:.0f}ms)"
        )

    # =================================================================
    # STIMULI
    # =================================================================
    def _setup_stimuli(self):
        self.digit_stim = visual.TextStim(
            self.win, text='', color='white', font='Arial', bold=True,
            pos=(0.0, 0.0), units='height')

        px = 2.0 if self.win.size[1] > 1200 else 1.0
        lw = 3.0 * px

        self.mask_circle = visual.Circle(
            self.win, radius=self.MASK_RADIUS, edges=64,
            lineColor='white', lineWidth=lw, fillColor=None, units='height')

        arm = self.MASK_RADIUS * 0.70
        self.mask_cross_a = visual.Line(
            self.win, start=(-arm, -arm), end=(arm, arm),
            lineColor='white', lineWidth=lw, units='height')
        self.mask_cross_b = visual.Line(
            self.win, start=(-arm, arm), end=(arm, -arm),
            lineColor='white', lineWidth=lw, units='height')

        self.fb_symbol = visual.TextStim(
            self.win, text='', height=0.14, pos=(0.0, 0.05), units='height')
        self.fb_msg = visual.TextStim(
            self.win, text='', color='white', height=0.045,
            pos=(0.0, -0.15), units='height')

    def _draw_digit(self, digit, font_size):
        self.digit_stim.text   = str(digit)
        self.digit_stim.height = font_size
        self.digit_stim.draw()

    def _draw_mask(self):
        self.mask_circle.draw()
        self.mask_cross_a.draw()
        self.mask_cross_b.draw()

    # =================================================================
    # FIXATION CROSS
    # =================================================================
    def _show_fixation_cross(self, duration=None):
        """Affiche le masque SART (cercle + croix diagonale) pendant `duration` secondes.

        Rendu frame-accurate. Le clavier est purgé en début de fonction afin
        d'éviter qu'un appui espace (écran d'instruction précédent) ne
        contamine le premier essai.
        Après la boucle, le back-buffer est propre : le premier flip de
        run_trial affiche directement le chiffre sans frame blanche parasite.
        """
        if duration is None:
            duration = self.FIXATION_DURATION_S

        self.flush_keyboard()
        n_frames = max(1, round(duration / self.frame_dur_s))

        self.logger.log(
            f"[SART] Masque-fixation début — {duration:.1f} s ({n_frames} frames)"
        )
        for _ in range(n_frames):
            self._draw_mask()          
            self.win.flip()
        self.logger.log("[SART] Masque-fixation fin")

    # =================================================================
    # INSTRUCTION SCREENS
    # =================================================================
    def _show_screen(self, text, show_mask=False, key='space'):
        stim = visual.TextStim(
            self.win, text=text, color='white',
            height=0.038, wrapWidth=1.3,
            pos=(0.0, 0.15 if show_mask else 0.0),
            units='height', alignText='center')
        stim.draw()

        if show_mask:
            arm   = self.MASK_RADIUS * 0.70
            y_off = -0.10
            circle = visual.Circle(
                self.win, radius=self.MASK_RADIUS, edges=64,
                lineColor='white', lineWidth=3.0, fillColor=None,
                pos=(0.0, y_off), units='height')
            cross_a = visual.Line(
                self.win, start=(-arm, -arm + y_off), end=(arm, arm + y_off),
                lineColor='white', lineWidth=3.0, units='height')
            cross_b = visual.Line(
                self.win, start=(-arm, arm + y_off), end=(arm, -arm + y_off),
                lineColor='white', lineWidth=3.0, units='height')
            circle.draw()
            cross_a.draw()
            cross_b.draw()

        prompt = visual.TextStim(
            self.win, text="Appuyez sur la barre d'espace pour continuer",
            color='grey', height=0.030, pos=(0.0, -0.40), units='height')
        prompt.draw()
        self.win.flip()
        self.wait_keys(key_list=[key])

    def show_instructions_screen1(self):
        txt = (
            "Dans cette étude, vous serez présenté une série de chiffres (1-9)\n"
            "de tailles variées au milieu de l'écran.\n\n"
            "Chaque chiffre est seulement présenté pendant une courte durée\n"
            "et est immédiatement couvert par un cercle croisé."
        )
        self._show_screen(txt, show_mask=True)

    def show_instructions_screen2(self):
        t = self.target_digit
        txt = (
            "TÂCHE :\n"
            f"Appuyez sur ESPACE lorsque vous voyez tout chiffre autre que {t}.\n"
            f"N'appuyez sur AUCUNE touche lorsque vous voyez le chiffre {t}.\n"
            "Attendez tout simplement pour le prochain chiffre.\n\n"
            "Soyez précis et rapide.\n"
            "Utilisez l'index de votre main dominante lorsque vous répondez\n"
            "(ex. : si vous êtes gaucher, utilisez votre index gauche\n"
            "pour appuyer sur ESPACE).\n\n"
            "Pratiquons cette tâche.\n\n"
            "Appuyez sur la barre d'espace pour COMMENCER"
        )
        self._show_screen(txt, show_mask=False)

    def show_instructions_screen3(self):
        t = self.target_digit
        txt = (
            "La pratique est terminée et la tâche complète va commencer.\n"
            "La tâche est la même que pendant la pratique, mais il n'y aura\n"
            "plus de retour d'information.\n\n"
            "Cela va prendre environ 4 minutes.\n\n"
            "TÂCHE :\n"
            f"Appuyez sur ESPACE lorsque vous voyez tout chiffre autre que {t}.\n"
            f"N'appuyez sur AUCUNE touche lorsque vous voyez le chiffre {t}.\n"
            "Attendez tout simplement pour le prochain chiffre.\n\n"
            "Soyez précis et rapide.\n"
            "Utilisez l'index de votre main dominante lorsque vous répondez.\n\n"
            "Appuyez sur la barre d'espace pour COMMENCER"
        )
        self._show_screen(txt, show_mask=False)

    # =================================================================
    # TRIAL LOADING — depuis Excel, filtré par colonne 'Block'
    # =================================================================
    def _parse_trials_from_df(self, df, block_label=''):
        """Parse un DataFrame pré-filtré en liste de dicts d'essai.

        Retourne (trials, n_go, n_nogo).
        Les lignes avec un digit invalide sont ignorées avec un warning
        (ex. : ligne 67 du bloc Main qui a TrialType vide mais IsNoGo=0).
        """
        trials = []
        for _, row in df.iterrows():
            # ── Digit ─────────────────────────────────────────────────────
            try:
                digit = int(row['digit'])
            except (ValueError, TypeError):
                self.logger.warn(
                    f"[SART] {block_label} — ligne ignorée, digit invalide : "
                    f"{row.get('digit')!r}"
                )
                continue

            is_nogo  = int(row['isnogo'])
            size_txt = str(row['size']).strip()

            pct_label, norm_height = self._size_text_to_info(size_txt)

            if norm_height not in [v['height'] for v in SIZE_MAP.values()]:
                self.logger.warn(
                    f"[SART] Taille inattendue '{size_txt}' → forçage 'medium'."
                )
                pct_label   = SIZE_MAP['medium']['pct']
                norm_height = SIZE_MAP['medium']['height']

            condition = 'nogo' if is_nogo == 1 else 'go'

            # TrialType peut être NaN (ex. : essai 67, bloc Main)
            trial_type_raw = str(row.get('trialtype', '')).strip()
            if trial_type_raw.upper() in ('NO-GO', 'NOGO'):
                trial_type_out = 'NoGo'
            elif trial_type_raw.upper() == 'GO':
                trial_type_out = 'Go'
            else:
                # Fallback basé sur IsNoGo
                trial_type_out = 'Go' if condition == 'go' else 'NoGo'

            trials.append({
                'trialnum':        int(row['trial']),
                'digit':           digit,
                'font_size_norm':  norm_height,
                'font_size_label': pct_label,
                'condition':       condition,
                'trial_type_out':  trial_type_out,
            })

        n_go   = sum(1 for t in trials if t['condition'] == 'go')
        n_nogo = sum(1 for t in trials if t['condition'] == 'nogo')
        self.logger.log(
            f"[SART] {block_label} chargé : "
            f"{len(trials)} essais ({n_go} GO, {n_nogo} NO-GO)"
        )
        return trials, n_go, n_nogo

    def load_trials_from_excel(self, block='Main'):
        """Charge les essais du bloc spécifié depuis SART_trials_McGill.xlsx.

        Parameters
        ----------
        block : str
            'Training' → 18 essais (colonne Block='Training')
            'Main'     → 225 essais (colonne Block='Main')

        Returns
        -------
        list[dict]
        """
        filepath = self.trial_file
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"[SART] Fichier introuvable : {filepath}")

        df = pd.read_excel(filepath)
        df.columns = [c.strip().lower() for c in df.columns]

        # Supprimer les lignes entièrement vides (lignes fantômes en bas de l'Excel)
        df = df.dropna(how='all')

        # Filtrer par colonne 'block' (insensible à la casse)
        mask_blk = (
            df['block'].astype(str).str.strip().str.lower()
            == block.strip().lower()
        )
        df_blk = df[mask_blk].copy()

        # Supprimer les lignes sans digit exploitable
        df_blk = df_blk.dropna(subset=['digit'])

        if df_blk.empty:
            self.logger.warn(
                f"[SART] ⚠️ Aucun essai pour le bloc '{block}' dans {filepath}"
            )
            return []

        trials, n_go, n_nogo = self._parse_trials_from_df(
            df_blk, block_label=block
        )

        # ── Validation des comptages attendus ─────────────────────────────
        key = block.strip().lower()
        if key == 'training':
            if len(trials) != TRAINING_N_TRIALS or n_nogo != TRAINING_N_NOGO:
                self.logger.warn(
                    f"[SART] ⚠️ Training : attendu {TRAINING_N_TRIALS} essais "
                    f"({TRAINING_N_GO} GO, {TRAINING_N_NOGO} NO-GO) — "
                    f"trouvé {len(trials)} ({n_go} GO, {n_nogo} NO-GO)"
                )
        elif key == 'main':
            if n_go != TEST_N_GO or n_nogo != TEST_N_NOGO:
                self.logger.warn(
                    f"[SART] ⚠️ Main : attendu {TEST_N_GO} GO + {TEST_N_NOGO} NO-GO — "
                    f"trouvé {n_go} GO + {n_nogo} NO-GO"
                )

        return trials

    # =================================================================
    # CORE TRIAL
    # =================================================================
    def run_trial(self, trial_index, total_trials, trial_info,
                  feedback=False, phase='test'):
        self.should_quit()
        gc.disable()

        digit          = trial_info['digit']
        font_size      = trial_info['font_size_norm']
        font_label     = trial_info['font_size_label']
        condition      = trial_info['condition']
        is_nogo        = (condition == 'nogo')
        trial_type_out = trial_info['trial_type_out']

        responded    = False
        response_key = None
        response_rt  = None

        self.flush_keyboard()

        # ── PHASE 1 : DIGIT (250 ms) ──────────────────────────────────────
        self._draw_digit(digit, font_size)
        t_digit_onset = self.win.flip()

        for _ in range(self.digit_n_frames - 1):
            if not responded:
                keys = self.get_keys([self.response_key])
                if keys:
                    responded    = True
                    response_key = keys[0].name
                    response_rt  = keys[0].tDown - t_digit_onset
            self._draw_digit(digit, font_size)
            self.win.flip()

        # ── PHASE 2 : MASK (900 ms) ───────────────────────────────────────
        self._draw_mask()
        t_mask_onset = self.win.flip()

        t_last_flip = t_mask_onset   # sécurité si mask_n_frames == 1

        for _ in range(self.mask_n_frames - 1):
            if not responded:
                keys = self.get_keys([self.response_key])
                if keys:
                    responded    = True
                    response_key = keys[0].name
                    response_rt  = keys[0].tDown - t_digit_onset
            self._draw_mask()
            t_last_flip = self.win.flip()

        t_mask_offset = t_last_flip

        # ── QC TIMING (séparé de l'Excel) ─────────────────────────────────
        actual_digit_ms = (t_mask_onset  - t_digit_onset) * 1000
        actual_mask_ms  = (t_mask_offset - t_mask_onset)  * 1000
        actual_total_ms = (t_mask_offset - t_digit_onset) * 1000

        timing_record = {
            'trial':             trial_index,
            'phase':             phase,
            'digit':             digit,
            'condition':         condition,
            'digit_onset':       round(t_digit_onset,  6),
            'mask_onset':        round(t_mask_onset,   6),
            'mask_offset':       round(t_mask_offset,  6),
            'actual_digit_ms':   round(actual_digit_ms, 2),
            'actual_mask_ms':    round(actual_mask_ms,  2),
            'actual_total_ms':   round(actual_total_ms, 2),
            'expected_digit_ms': 250.0,
            'expected_mask_ms':  900.0,
            'expected_total_ms': 1150.0,
            'digit_error_ms':    round(actual_digit_ms - 250.0,  2),
            'mask_error_ms':     round(actual_mask_ms  - 900.0,  2),
            'total_error_ms':    round(actual_total_ms - 1150.0, 2),
        }
        self.timing_log.append(timing_record)

        threshold_ms = self.frame_dur_s * 1000 * 1.5
        if abs(timing_record['digit_error_ms']) > threshold_ms:
            self.logger.warn(
                f"[QC] Trial {trial_index}: digit={actual_digit_ms:.1f}ms "
                f"(err={timing_record['digit_error_ms']:+.1f}ms)"
            )
        if abs(timing_record['mask_error_ms']) > threshold_ms:
            self.logger.warn(
                f"[QC] Trial {trial_index}: mask={actual_mask_ms:.1f}ms "
                f"(err={timing_record['mask_error_ms']:+.1f}ms)"
            )

        # ── RT ET LATENCYTYPE ─────────────────────────────────────────────
        if responded:
            response_code = self._key_to_code(response_key)
            rt_ms         = round(response_rt * 1000)
            latency       = rt_ms
            if rt_ms < RT_ANTICIPATORY_MAX:
                latency_type = 1
            elif rt_ms < RT_AMBIGUOUS_MAX:
                latency_type = 2
            else:
                latency_type = 3
        else:
            response_code = 0
            rt_ms         = ''
            latency       = TOTAL_TRIAL_MS
            latency_type  = 0

        # ── CLASSIFICATION ────────────────────────────────────────────────
        if is_nogo:
            if responded:
                accuracy = 'NoGo Failure'
                corr     = 0
                self.perf['nogo_commission'] += 1
            else:
                accuracy = 'NoGo Success'
                corr     = 1
                self.perf['nogo_correct'] += 1
        else:
            if responded:
                if latency_type == 1:
                    accuracy = 'Go Anticipatory'
                elif latency_type == 2:
                    accuracy = 'Go Ambiguous'
                else:
                    accuracy = 'Go Success'
                corr = 1
                self.perf['go_correct'] += 1
                self.perf['go_rts'].append(response_rt)
            else:
                accuracy = 'Omission'
                corr     = 0
                self.perf['go_omission'] += 1

        # ── COMPTEURS CUMULATIFS (phase test uniquement) ───────────────────
        if phase == 'test':
            if not is_nogo:
                self.perf['count_go'] += 1
                if latency_type == 1:
                    self.perf['count_anticipatory'] += 1
                elif latency_type == 0:
                    self.perf['incorrect_suppressions'] += 1
                elif latency_type == 3:
                    self.perf['count_valid_go'] += 1
            else:
                self.perf['count_nogo'] += 1
                if not responded:
                    self.perf['correct_suppressions'] += 1

        # ── Validation RT ─────────────────────────────────────────────────
        if responded and isinstance(rt_ms, int) and \
                (rt_ms < 0 or rt_ms > TOTAL_TRIAL_MS + 100):
            self.logger.warn(
                f"[SART] ⚠️ RT aberrant trial {trial_index}: {rt_ms}ms"
            )

        # ── FEEDBACK (entraînement, si activé) ────────────────────────────
        if feedback:
            if corr == 1:
                self.fb_symbol.text, self.fb_symbol.color = '✓', 'green'
                if accuracy in ('Go Success', 'Go Ambiguous', 'Go Anticipatory'):
                    self.fb_msg.text = f"Correct ! RT : {rt_ms} ms"
                else:
                    self.fb_msg.text = "Correct ! Bonne inhibition ✓"
            else:
                self.fb_symbol.text, self.fb_symbol.color = '✗', 'red'
                if accuracy == 'NoGo Failure':
                    self.fb_msg.text = (
                        f"Erreur — ne PAS appuyer pour le {self.target_digit}"
                    )
                else:
                    self.fb_msg.text = "Erreur — appuyez pour les autres chiffres"
            self.fb_symbol.draw()
            self.fb_msg.draw()
            self.win.flip()
            core.wait(self.feedback_dur)

        # ── ENREGISTREMENT ────────────────────────────────────────────────
        cnt_ant  = self.perf['count_anticipatory']      if phase == 'test' else 0
        cnt_cs   = self.perf['correct_suppressions']    if phase == 'test' else 0
        cnt_is   = self.perf['incorrect_suppressions']  if phase == 'test' else 0
        cnt_nogo = self.perf['count_nogo']              if phase == 'test' else 0
        cnt_go   = self.perf['count_go']                if phase == 'test' else 0
        cnt_vgo  = self.perf['count_valid_go']          if phase == 'test' else 0

        record = {
            'phase':                   phase,
            'trialCount':              trial_index - 1,
            'digitPresentationTime':   250,
            'maskPresentationTime':    900,
            'trialType':               trial_type_out,
            'digit':                   digit,
            'fontSize':                font_label,
            'response':                response_code,
            'correct':                 corr,
            'rt':                      rt_ms,
            'latency':                 latency,
            'latencyType':             latency_type,
            'responseType':            accuracy,
            'countAnticipatory':       cnt_ant,
            'correctSuppressions':     cnt_cs,
            'incorrectSuppressions':   cnt_is,
            'countNoGo':               cnt_nogo,
            'countGo':                 cnt_go,
            'countValidGo':            cnt_vgo,
            'countProbes':             0,
            'radioButtons.difficulty.response': '',
            'radioButtons.interest.response':   '',
        }
        self.trial_data.append(record)
        self.save_trial_incremental(record)

        rt_str = f"{rt_ms}ms" if responded else "  ---  "
        print(
            f"  {trial_index:>3}/{total_trials:<3} | d={digit} "
            f"{'NOGO' if is_nogo else ' GO '} | "
            f"Resp={'Y' if responded else 'N'} | RT={rt_str:>8} | {accuracy}"
        )

        if trial_index % 50 == 0:
            gc.enable()
            gc.collect()
            gc.disable()

        return record

    # =================================================================
    # BLOCK & METRICS
    # =================================================================
    def run_block(self, trials, block_name, feedback=False, phase='test'):
        total = len(trials)
        print(
            f"\n{'='*55}\n  Block : {block_name} | {total} essais | "
            f"feedback={'ON' if feedback else 'OFF'}\n{'='*55}"
        )
        for i, trial_info in enumerate(trials, start=1):
            self.run_trial(
                trial_index=i, total_trials=total,
                trial_info=trial_info, feedback=feedback,
                phase=phase,
            )
        self._print_performance(block_name)

    def compute_metrics(self):
        p          = self.perf
        total_go   = p['go_correct'] + p['go_omission']
        total_nogo = p['nogo_correct'] + p['nogo_commission']

        go_acc   = (p['go_correct']  / total_go   * 100) if total_go   else 0.0
        nogo_acc = (p['nogo_correct'] / total_nogo * 100) if total_nogo else 0.0

        if p['go_rts']:
            rts       = np.array(p['go_rts']) * 1000
            mean_rt   = float(np.mean(rts))
            median_rt = float(np.median(rts))
        else:
            mean_rt = median_rt = 0.0

        return {
            'total_go': total_go, 'total_nogo': total_nogo,
            'go_correct': p['go_correct'], 'go_omission': p['go_omission'],
            'nogo_correct': p['nogo_correct'], 'nogo_commission': p['nogo_commission'],
            'go_accuracy_pct': round(go_acc, 1), 'nogo_accuracy_pct': round(nogo_acc, 1),
            'mean_rt_ms': round(mean_rt, 1), 'median_rt_ms': round(median_rt, 1),
        }

    def _print_performance(self, block_name=""):
        m = self.compute_metrics()
        print(
            f"\n{'─'*55}\n  SART Performance — {block_name}\n{'─'*55}\n"
            f"  GO  : {m['go_accuracy_pct']:5.1f}% corrects  "
            f"({m['go_correct']}/{m['total_go']})  |  omissions : {m['go_omission']}\n"
            f"  NOGO: {m['nogo_accuracy_pct']:5.1f}% corrects  "
            f"({m['nogo_correct']}/{m['total_nogo']})  |  commissions : {m['nogo_commission']}\n"
            f"  RT moyen : {m['mean_rt_ms']:6.1f} ms  |  RT médian : {m['median_rt_ms']:6.1f} ms\n"
            f"{'─'*55}"
        )

    # =================================================================
    # SAVE — Training + Test dans le même fichier Excel (3 feuilles)
    # =================================================================
    def save_data(self, **kwargs):
        has_training = len(self.training_data) > 0
        has_test     = len(self.test_data) > 0

        if not has_training and not has_test:
            self.logger.warn("[SART] Aucune donnée à sauvegarder.")
            return None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename  = (
            f"McGill_SART_Raw_Data_{self.nom}_ses-{self.session}"
            f"_{self.mode}_{timestamp}.xlsx"
        )
        filepath = os.path.join(self.data_dir, filename)

        col_order = [
            'phase', 'trialCount', 'digitPresentationTime', 'maskPresentationTime',
            'trialType', 'digit', 'fontSize', 'response', 'correct', 'rt',
            'latency', 'latencyType', 'responseType',
            'countAnticipatory', 'correctSuppressions', 'incorrectSuppressions',
            'countNoGo', 'countGo', 'countValidGo', 'countProbes',
            'radioButtons.difficulty.response', 'radioButtons.interest.response',
        ]

        def _to_df(data):
            df = pd.DataFrame(data)
            return df[[c for c in col_order if c in df.columns]]

        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                if has_training:
                    _to_df(self.training_data).to_excel(
                        writer, sheet_name='Training', index=False)
                    self.logger.log(
                        f"[SART] Training : {len(self.training_data)} essais → feuille 'Training'"
                    )
                if has_test:
                    _to_df(self.test_data).to_excel(
                        writer, sheet_name='Test', index=False)
                    self.logger.log(
                        f"[SART] Test : {len(self.test_data)} essais → feuille 'Test'"
                    )
                # Feuille combinée Training + Test (dans l'ordre chronologique)
                all_data = self.training_data + self.test_data
                if all_data:
                    _to_df(all_data).to_excel(
                        writer, sheet_name='All_Trials', index=False)
                    self.logger.log(
                        f"[SART] All_Trials : {len(all_data)} essais → feuille 'All_Trials'"
                    )

            self.logger.ok(f"[SART] Données → {filepath}")

        except Exception as e:
            self.logger.warn(f"[SART] Erreur sauvegarde : {e}")
            csv_path = filepath.replace('.xlsx', '_EMERGENCY.csv')
            pd.DataFrame(self.training_data + self.test_data).to_csv(
                csv_path, index=False)
            self.logger.warn(f"[SART] Fallback CSV → {csv_path}")
            filepath = csv_path

        return filepath

    # =================================================================
    # MAIN RUN — séquence explicite par mode
    # =================================================================
    def run(self):
        """
        Séquences selon le mode :

        full          : screen1 → screen2 → fixation(2s) → training(18)
                        → screen3 → fixation(2s) → test(225)

        training_only : screen1 → screen2 → fixation(2s) → training(18)

        test_only     : screen1 → screen2 → fixation(2s) → test(225)
        """
        filepath = None
        aborted  = False

        try:
            # ── Écran 1 : toujours affiché en premier ─────────────────────
            self.show_instructions_screen1()

            self.logger.log(
                f"[SART] run() | mode={self.mode} | "
                f"training_feedback={self.training_feedback}"
            )

            # ══════════════════════════════════════════════════════════════
            # MODE FULL
            # screen2 → fixation(2s) → training(18)
            # screen3 → fixation(2s) → test(225)
            # ══════════════════════════════════════════════════════════════
            if self.mode == 'full':

                # ── Entraînement ──────────────────────────────────────────
                self.show_instructions_screen2()
                self._show_fixation_cross()                  # ← 2 s
                self._set_phase('training')
                self.perf = self._empty_perf()
                self.task_clock.reset()
                self.run_block(
                    self.load_trials_from_excel(block='Training'),
                    block_name="TRAINING",
                    feedback=self.training_feedback,
                    phase='training',
                )

                # ── Test ──────────────────────────────────────────────────
                self.show_instructions_screen3()
                self._show_fixation_cross()                  # ← 2 s
                self._set_phase('test')
                self.perf = self._empty_perf()
                self.task_clock.reset()
                self.run_block(
                    self.load_trials_from_excel(block='Main'),
                    block_name="TEST",
                    feedback=False,
                    phase='test',
                )

            # ══════════════════════════════════════════════════════════════
            # MODE TRAINING ONLY
            # screen2 → fixation(2s) → training(18)
            # ══════════════════════════════════════════════════════════════
            elif self.mode == 'training_only':

                self.show_instructions_screen2()
                self._show_fixation_cross()                  # ← 2 s
                self._set_phase('training')
                self.perf = self._empty_perf()
                self.task_clock.reset()
                self.run_block(
                    self.load_trials_from_excel(block='Training'),
                    block_name="TRAINING",
                    feedback=self.training_feedback,
                    phase='training',
                )

            # ══════════════════════════════════════════════════════════════
            # MODE TEST ONLY
            # screen2 → fixation(2s) → test(225)
            # ══════════════════════════════════════════════════════════════
            elif self.mode == 'test_only':

                self.show_instructions_screen2()
                self._show_fixation_cross()                  # ← 2 s
                self._set_phase('test')
                self.perf = self._empty_perf()
                self.task_clock.reset()
                self.run_block(
                    self.load_trials_from_excel(block='Main'),
                    block_name="TEST",
                    feedback=False,
                    phase='test',
                )

            else:
                raise ValueError(f"[SART] Mode non reconnu : '{self.mode}'")

            self.logger.ok("[SART] Tâche terminée ✓")

        except (KeyboardInterrupt, SystemExit):
            self.logger.warn("[SART] Interruption volontaire détectée.")
            aborted = True

        except Exception as e:
            self.logger.warn(f"[SART] Erreur : {e}")
            import traceback
            traceback.print_exc()
            aborted = True

        finally:
            filepath = self.save_data()
            self._print_performance("FINAL")

            qc_timing = self.test_timing if self.test_timing else self.training_timing
            qc_label  = 'test' if self.test_timing else 'training'

            if self.doqc:
                from tasks.qc.qc_sart import SARTTimingQC
                qc = SARTTimingQC(
                    timing_log=qc_timing,
                    frame_rate=self.frame_rate,
                    frame_dur_s=self.frame_dur_s,
                    digit_n_frames=self.digit_n_frames,
                    mask_n_frames=self.mask_n_frames,
                    participant_id=self.nom,
                    session=self.session,
                    data_dir=self.data_dir,
                    logger=self.logger,
                )
                self.logger.log(
                    f"[SART] QC sur phase '{qc_label}' ({len(qc_timing)} essais)"
                )
                qc.run_qc()

            if not aborted:
                self._show_screen(
                    "Fin de la tâche.\n\nMerci pour votre participation.",
                    show_mask=False,
                )

        return filepath