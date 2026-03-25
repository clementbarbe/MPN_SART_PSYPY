"""
sart.py — Sustained Attention to Response Task (McGill Protocol)
================================================================
Modes :
    'full'      → screen1 → screen2 → fixation(2s) → 18 essais training
                  → screen3 → fixation(2s) → 225 essais test
    'training'  → screen1 → screen2 → fixation(2s) → 18 essais training
    'test'      → screen1 → screen2 → fixation(2s) → 225 essais test

Paramètre 'training_feedback' (bool, défaut True) :
    True  → feedback visuel après chaque essai d'entraînement

Classification latencyType :
    0 → aucune réponse  (latency = 1150 ms)
    1 → RT < 100 ms     → Go Anticipatory / NoGo Failure
    2 → 100 ≤ RT < 200  → Go Ambiguous   / NoGo Failure
    3 → RT ≥ 200 ms     → Go Success     / NoGo Failure

Entrée : SART_trials_McGill.xlsx
    · Block='Training' → 18 essais (16 GO, 2 NO-GO)
    · Block='Main'     → 225 essais (200 GO, 25 NO-GO)
Sortie : McGill_SART_Raw_Data_*.xlsx (feuille unique : All_Trials)
"""

import gc
import os
import traceback
import numpy as np
import pandas as pd
from datetime import datetime

from psychopy import visual, core
from utils.base_task import BaseTask


# =========================================================================
# CONFIG
# =========================================================================
DEFAULT_CONFIG = {
    'mode':              'full',
    'participant_id':    'P001',
    'session':           '01',
    'target_digit':      3,
    'response_key':      'space',
    'trial_file':        'SART_trials_McGill.xlsx',
    'data_dir':          'data/sart',
    'feedback_duration': 0.800,
    'training_feedback': True,
}

MODE_ALIASES = {
    'full':     'full',
    'training': 'training_only',
    'test':     'test_only',
}

SIZE_MAP = {
    'tiny':       {'pct': '2pct',  'height': 0.04},
    'small':      {'pct': '4pct',  'height': 0.06},
    'medium':     {'pct': '10pct', 'height': 0.10},
    'large':      {'pct': '16pct', 'height': 0.14},
    'very large': {'pct': '16pct', 'height': 0.18},
}

KEY_CODE_MAP        = {'space': 57}
TOTAL_TRIAL_MS      = 1150
RT_ANTICIPATORY_MAX = 100
RT_AMBIGUOUS_MAX    = 200

EXPECTED = {
    'training': {'trials': 18, 'go': 16, 'nogo': 2},
    'main':     {'trials': 225, 'go': 200, 'nogo': 25},
}

COL_ORDER = [
    'phase', 'trialCount', 'digitPresentationTime', 'maskPresentationTime',
    'trialType', 'digit', 'fontSize', 'response', 'correct', 'rt',
    'latency', 'latencyType', 'responseType',
    'countAnticipatory', 'correctSuppressions', 'incorrectSuppressions',
    'countNoGo', 'countGo', 'countValidGo', 'countProbes',
    'radioButtons.difficulty.response', 'radioButtons.interest.response',
]


# =========================================================================
# TASK
# =========================================================================
class sart(BaseTask):
    """Sustained Attention to Response Task — McGill Protocol."""

    DIGIT_DURATION_S    = 0.250
    MASK_DURATION_S     = 0.900
    MASK_RADIUS         = 0.08
    FIXATION_DURATION_S = 2.0

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------
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

        raw_mode  = cfg['mode'].lower().strip()
        self.mode = MODE_ALIASES.get(raw_mode, 'full')
        if self.mode == 'full' and raw_mode not in MODE_ALIASES:
            self.logger.warn(
                f"[SART] Mode '{raw_mode}' inconnu → fallback 'full'. "
                f"Valides : {sorted(MODE_ALIASES)}"
            )

        self.target_digit      = int(cfg['target_digit'])
        self.response_key      = cfg['response_key']
        self.trial_file        = cfg['trial_file']
        self.feedback_dur      = float(cfg['feedback_duration'])
        self.training_feedback = bool(cfg.get('training_feedback', True))

        self.training_data   = []
        self.test_data       = []
        self.training_timing = []
        self.test_timing     = []

        # Pointeurs actifs (basculés par _set_phase)
        self.trial_data = self.training_data
        self.timing_log = self.training_timing

        self.perf  = self._empty_perf()
        self.doqc  = False

        self._measure_frame_rate()
        self._setup_stimuli()

        self.logger.log(
            f"[SART] mode={self.mode} | target={self.target_digit} | "
            f"training_feedback={self.training_feedback} | "
            f"fps={self.frame_rate:.1f} | frame={self.frame_dur_s*1000:.2f}ms"
        )

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    @staticmethod
    def _empty_perf():
        return {
            'go_correct': 0, 'go_omission': 0,
            'nogo_correct': 0, 'nogo_commission': 0,
            'go_rts': [],
            'count_anticipatory': 0, 'correct_suppressions': 0,
            'incorrect_suppressions': 0,
            'count_nogo': 0, 'count_go': 0, 'count_valid_go': 0,
        }

    def _set_phase(self, phase):
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
    def _size_to_info(size_text):
        key  = str(size_text).strip().lower()
        info = SIZE_MAP.get(key)
        if info:
            return info['pct'], info['height']
        try:
            val = float(size_text)
            return f"{int(val*100)}pct", val
        except (ValueError, TypeError):
            return SIZE_MAP['medium']['pct'], SIZE_MAP['medium']['height']

    @staticmethod
    def _key_to_code(key_name):
        if not key_name:
            return 0
        return KEY_CODE_MAP.get(key_name, ord(key_name[0]))

    def _measure_frame_rate(self):
        measured = self.win.getActualFrameRate(nIdentical=10, nMaxFrames=100, threshold=1)
        self.frame_rate        = measured if measured else 60.0
        self.frame_dur_s       = 1.0 / self.frame_rate
        self.digit_n_frames    = max(1, round(self.DIGIT_DURATION_S    / self.frame_dur_s))
        self.mask_n_frames     = max(1, round(self.MASK_DURATION_S     / self.frame_dur_s))
        self.fixation_n_frames = max(1, round(self.FIXATION_DURATION_S / self.frame_dur_s))

    # ------------------------------------------------------------------
    # STIMULI
    # ------------------------------------------------------------------
    def _setup_stimuli(self):
        self.digit_stim = visual.TextStim(
            self.win, text='', color='white', font='Arial', bold=True,
            pos=(0, 0), units='height')

        lw = 3.0 * (2.0 if self.win.size[1] > 1200 else 1.0)
        arm = self.MASK_RADIUS * 0.70

        self.mask_circle  = visual.Circle(
            self.win, radius=self.MASK_RADIUS, edges=64,
            lineColor='white', lineWidth=lw, fillColor=None, units='height')
        self.mask_cross_a = visual.Line(
            self.win, start=(-arm, -arm), end=(arm, arm),
            lineColor='white', lineWidth=lw, units='height')
        self.mask_cross_b = visual.Line(
            self.win, start=(-arm, arm), end=(arm, -arm),
            lineColor='white', lineWidth=lw, units='height')

        self.fb_symbol = visual.TextStim(
            self.win, text='', height=0.14, pos=(0, 0.05), units='height')
        self.fb_msg = visual.TextStim(
            self.win, text='', color='white', height=0.045,
            pos=(0, -0.15), units='height')

    def _draw_digit(self, digit, size):
        self.digit_stim.text   = str(digit)
        self.digit_stim.height = size
        self.digit_stim.draw()

    def _draw_mask(self):
        self.mask_circle.draw()
        self.mask_cross_a.draw()
        self.mask_cross_b.draw()

    # ------------------------------------------------------------------
    # FIXATION (masque SART affiché 2 s avant chaque bloc)
    # ------------------------------------------------------------------
    def _show_fixation(self, duration=None):
        """Affiche le masque SART comme fixation pendant `duration` secondes."""
        n = max(1, round((duration or self.FIXATION_DURATION_S) / self.frame_dur_s))
        self.flush_keyboard()
        for _ in range(n):
            self._draw_mask()
            self.win.flip()

    # ------------------------------------------------------------------
    # INSTRUCTION SCREENS
    # ------------------------------------------------------------------
    def _show_screen(self, text, show_mask=False, key='space'):
        pos = (0.0, 0.15) if show_mask else (0.0, 0.0)
        visual.TextStim(
            self.win, text=text, color='white',
            height=0.038, wrapWidth=1.3,
            pos=pos, units='height', alignText='center').draw()

        if show_mask:
            arm, y = self.MASK_RADIUS * 0.70, -0.10
            visual.Circle(
                self.win, radius=self.MASK_RADIUS, edges=64,
                lineColor='white', lineWidth=3.0, fillColor=None,
                pos=(0, y), units='height').draw()
            visual.Line(
                self.win, start=(-arm, -arm + y), end=(arm, arm + y),
                lineColor='white', lineWidth=3.0, units='height').draw()
            visual.Line(
                self.win, start=(-arm, arm + y), end=(arm, -arm + y),
                lineColor='white', lineWidth=3.0, units='height').draw()

        visual.TextStim(
            self.win, text="Appuyez sur la barre d'espace pour continuer",
            color='grey', height=0.030, pos=(0, -0.40), units='height').draw()

        self.win.flip()
        self.wait_keys(key_list=[key])

    def _screen1(self):
        self._show_screen(
            "Dans cette étude, vous serez présenté une série de chiffres (1-9)\n"
            "de tailles variées au milieu de l'écran.\n\n"
            "Chaque chiffre est seulement présenté pendant une courte durée\n"
            "et est immédiatement couvert par un cercle croisé.",
            show_mask=True
        )

    def _screen2(self):
        t = self.target_digit
        self._show_screen(
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

    def _screen3(self):
        t = self.target_digit
        self._show_screen(
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

    # ------------------------------------------------------------------
    # TRIAL LOADING
    # ------------------------------------------------------------------
    def load_trials(self, block):
        """Charge les essais d'un bloc depuis l'Excel (block='Training' ou 'Main')."""
        if not os.path.isfile(self.trial_file):
            raise FileNotFoundError(f"[SART] Fichier introuvable : {self.trial_file}")

        df = pd.read_excel(self.trial_file)
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.dropna(how='all')

        df_blk = df[
            df['block'].astype(str).str.strip().str.lower() == block.lower()
        ].dropna(subset=['digit']).copy()

        if df_blk.empty:
            self.logger.warn(f"[SART] Aucun essai pour le bloc '{block}'.")
            return []

        trials = []
        for _, row in df_blk.iterrows():
            try:
                digit = int(row['digit'])
            except (ValueError, TypeError):
                self.logger.warn(f"[SART] Ligne ignorée, digit invalide : {row.get('digit')!r}")
                continue

            is_nogo   = int(row['isnogo'])
            condition = 'nogo' if is_nogo else 'go'
            pct, h    = self._size_to_info(row['size'])

            if h not in [v['height'] for v in SIZE_MAP.values()]:
                self.logger.warn(f"[SART] Taille '{row['size']}' inconnue → 'medium'.")
                pct, h = SIZE_MAP['medium']['pct'], SIZE_MAP['medium']['height']

            raw_type = str(row.get('trialtype', '')).strip().upper()
            if raw_type in ('NO-GO', 'NOGO'):
                trial_type = 'NoGo'
            elif raw_type == 'GO':
                trial_type = 'Go'
            else:
                trial_type = 'Go' if condition == 'go' else 'NoGo'

            trials.append({
                'trialnum':       int(row['trial']),
                'digit':          digit,
                'font_size_norm': h,
                'font_size_label': pct,
                'condition':      condition,
                'trial_type_out': trial_type,
            })

        n_go   = sum(1 for t in trials if t['condition'] == 'go')
        n_nogo = sum(1 for t in trials if t['condition'] == 'nogo')
        exp    = EXPECTED.get(block.lower(), {})
        self.logger.log(
            f"[SART] Bloc '{block}' : {len(trials)} essais ({n_go} GO, {n_nogo} NO-GO)"
        )
        if exp and (n_go != exp['go'] or n_nogo != exp['nogo']):
            self.logger.warn(
                f"[SART] Attendu {exp['go']} GO + {exp['nogo']} NO-GO, "
                f"trouvé {n_go} GO + {n_nogo} NO-GO."
            )
        return trials

    # ------------------------------------------------------------------
    # CORE TRIAL
    # ------------------------------------------------------------------
    def run_trial(self, trial_index, total_trials, trial_info,
                  feedback=False, phase='test'):
        self.should_quit()
        gc.disable()

        digit     = trial_info['digit']
        font_size = trial_info['font_size_norm']
        font_lbl  = trial_info['font_size_label']
        is_nogo   = trial_info['condition'] == 'nogo'
        ttype     = trial_info['trial_type_out']

        responded    = False
        response_key = None
        response_rt  = None

        self.flush_keyboard()

        # ── Digit (250 ms) ────────────────────────────────────────────────
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

        # ── Masque (900 ms) ───────────────────────────────────────────────
        self._draw_mask()
        t_mask_onset = self.win.flip()
        t_last_flip  = t_mask_onset

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

        # ── QC Timing ────────────────────────────────────────────────────
        actual_digit_ms = (t_mask_onset  - t_digit_onset) * 1000
        actual_mask_ms  = (t_mask_offset - t_mask_onset)  * 1000
        actual_total_ms = (t_mask_offset - t_digit_onset) * 1000
        thr = self.frame_dur_s * 1000 * 1.5

        self.timing_log.append({
            'trial': trial_index, 'phase': phase,
            'digit': digit, 'condition': trial_info['condition'],
            'actual_digit_ms': round(actual_digit_ms, 2),
            'actual_mask_ms':  round(actual_mask_ms,  2),
            'actual_total_ms': round(actual_total_ms, 2),
            'digit_error_ms':  round(actual_digit_ms - 250.0,  2),
            'mask_error_ms':   round(actual_mask_ms  - 900.0,  2),
            'total_error_ms':  round(actual_total_ms - 1150.0, 2),
        })

        for label, err in (('digit', actual_digit_ms - 250), ('mask', actual_mask_ms - 900)):
            if abs(err) > thr:
                self.logger.warn(f"[QC] Trial {trial_index} {label} err={err:+.1f}ms")

        # ── RT & latencyType ─────────────────────────────────────────────
        if responded:
            rt_ms        = round(response_rt * 1000)
            latency      = rt_ms
            latency_type = 1 if rt_ms < RT_ANTICIPATORY_MAX else (2 if rt_ms < RT_AMBIGUOUS_MAX else 3)
            resp_code    = self._key_to_code(response_key)
        else:
            rt_ms        = ''
            latency      = TOTAL_TRIAL_MS
            latency_type = 0
            resp_code    = 0

        # ── Classification ────────────────────────────────────────────────
        if is_nogo:
            if responded:
                accuracy, corr = 'NoGo Failure', 0
                self.perf['nogo_commission'] += 1
            else:
                accuracy, corr = 'NoGo Success', 1
                self.perf['nogo_correct'] += 1
        else:
            if responded:
                accuracy = ('Go Anticipatory' if latency_type == 1
                            else 'Go Ambiguous' if latency_type == 2
                            else 'Go Success')
                corr = 1
                self.perf['go_correct'] += 1
                self.perf['go_rts'].append(response_rt)
            else:
                accuracy, corr = 'Omission', 0
                self.perf['go_omission'] += 1

        # ── Compteurs cumulatifs (test uniquement) ────────────────────────
        if phase == 'test':
            if not is_nogo:
                self.perf['count_go'] += 1
                if   latency_type == 1: self.perf['count_anticipatory']    += 1
                elif latency_type == 0: self.perf['incorrect_suppressions'] += 1
                elif latency_type == 3: self.perf['count_valid_go']         += 1
            else:
                self.perf['count_nogo'] += 1
                if not responded:
                    self.perf['correct_suppressions'] += 1

        # ── Feedback (entraînement) ───────────────────────────────────────
        if feedback:
            if corr:
                self.fb_symbol.text, self.fb_symbol.color = '✓', 'green'
                self.fb_msg.text = (
                    f"Correct ! RT : {rt_ms} ms"
                    if accuracy in ('Go Success', 'Go Ambiguous', 'Go Anticipatory')
                    else "Correct ! Bonne inhibition ✓"
                )
            else:
                self.fb_symbol.text, self.fb_symbol.color = '✗', 'red'
                self.fb_msg.text = (
                    f"Erreur — ne PAS appuyer pour le {self.target_digit}"
                    if accuracy == 'NoGo Failure'
                    else "Erreur — appuyez pour les autres chiffres"
                )
            self.fb_symbol.draw()
            self.fb_msg.draw()
            self.win.flip()
            core.wait(self.feedback_dur)

        # ── Enregistrement ────────────────────────────────────────────────
        p = self.perf
        record = {
            'phase':                   phase,
            'trialCount':              trial_index - 1,
            'digitPresentationTime':   250,
            'maskPresentationTime':    900,
            'trialType':               ttype,
            'digit':                   digit,
            'fontSize':                font_lbl,
            'response':                resp_code,
            'correct':                 corr,
            'rt':                      rt_ms,
            'latency':                 latency,
            'latencyType':             latency_type,
            'responseType':            accuracy,
            'countAnticipatory':       p['count_anticipatory']     if phase == 'test' else 0,
            'correctSuppressions':     p['correct_suppressions']   if phase == 'test' else 0,
            'incorrectSuppressions':   p['incorrect_suppressions'] if phase == 'test' else 0,
            'countNoGo':               p['count_nogo']             if phase == 'test' else 0,
            'countGo':                 p['count_go']               if phase == 'test' else 0,
            'countValidGo':            p['count_valid_go']         if phase == 'test' else 0,
            'countProbes':             0,
            'radioButtons.difficulty.response': '',
            'radioButtons.interest.response':   '',
        }
        self.trial_data.append(record)
        self.save_trial_incremental(record)

        rt_str = f"{rt_ms}ms" if responded else "---"
        print(f"  {trial_index:>3}/{total_trials} | "
              f"d={digit} {'NOGO' if is_nogo else ' GO '} | "
              f"Resp={'Y' if responded else 'N'} | RT={rt_str:>7} | {accuracy}")

        if trial_index % 50 == 0:
            gc.enable(); gc.collect(); gc.disable()

        return record

    # ------------------------------------------------------------------
    # BLOCK
    # ------------------------------------------------------------------
    def run_block(self, trials, block_name, feedback=False, phase='test'):
        total = len(trials)
        print(f"\n{'='*55}\n  {block_name} | {total} essais | "
              f"feedback={'ON' if feedback else 'OFF'}\n{'='*55}")
        for i, t in enumerate(trials, start=1):
            self.run_trial(i, total, t, feedback=feedback, phase=phase)
        self._print_perf(block_name)

    # ------------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------------
    def _print_perf(self, label=""):
        p  = self.perf
        tg = p['go_correct'] + p['go_omission']
        tn = p['nogo_correct'] + p['nogo_commission']
        ga = p['go_correct']   / tg * 100 if tg else 0
        na = p['nogo_correct'] / tn * 100 if tn else 0
        rts = np.array(p['go_rts']) * 1000 if p['go_rts'] else np.array([0.0])
        print(
            f"\n{'─'*55}\n  Performance — {label}\n{'─'*55}\n"
            f"  GO  : {ga:5.1f}% ({p['go_correct']}/{tg}) | omissions : {p['go_omission']}\n"
            f"  NOGO: {na:5.1f}% ({p['nogo_correct']}/{tn}) | commissions : {p['nogo_commission']}\n"
            f"  RT  : moy={np.mean(rts):.1f}ms | méd={np.median(rts):.1f}ms\n"
            f"{'─'*55}"
        )

    # ------------------------------------------------------------------
    # SAVE — feuille unique All_Trials
    # ------------------------------------------------------------------
    def save_data(self, **kwargs):
        all_data = self.training_data + self.test_data
        if not all_data:
            self.logger.warn("[SART] Aucune donnée à sauvegarder.")
            return None

        ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = (f"McGill_SART_Raw_Data_{self.nom}_ses-{self.session}"
                    f"_{self.mode}_{ts}.xlsx")
        filepath = os.path.join(self.data_dir, filename)

        df = pd.DataFrame(all_data)
        df = df[[c for c in COL_ORDER if c in df.columns]]

        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='All_Trials', index=False)
            self.logger.ok(f"[SART] {len(all_data)} essais → {filepath}")

        except Exception as e:
            self.logger.warn(f"[SART] Erreur sauvegarde : {e}")
            filepath = filepath.replace('.xlsx', '_EMERGENCY.csv')
            df.to_csv(filepath, index=False)
            self.logger.warn(f"[SART] Fallback CSV → {filepath}")

        return filepath

    # ------------------------------------------------------------------
    # RUN
    # ------------------------------------------------------------------
    def run(self):
        """
        full          : screen1 → screen2 → fixation → training
                        → screen3 → fixation → test
        training_only : screen1 → screen2 → fixation → training
        test_only     : screen1 → screen2 → fixation → test
        """
        filepath = None
        aborted  = False

        try:
            self._screen1()
            self.logger.log(f"[SART] run() | mode={self.mode}")

            if self.mode == 'full':
                self._screen2()
                self._show_fixation()
                self._set_phase('training')
                self.perf = self._empty_perf()
                self.task_clock.reset()
                self.run_block(
                    self.load_trials('Training'),
                    block_name="TRAINING",
                    feedback=self.training_feedback,
                    phase='training',
                )
                self._screen3()
                self._show_fixation()
                self._set_phase('test')
                self.perf = self._empty_perf()
                self.task_clock.reset()
                self.run_block(
                    self.load_trials('Main'),
                    block_name="TEST",
                    feedback=False,
                    phase='test',
                )

            elif self.mode == 'training_only':
                self._screen2()
                self._show_fixation()
                self._set_phase('training')
                self.perf = self._empty_perf()
                self.task_clock.reset()
                self.run_block(
                    self.load_trials('Training'),
                    block_name="TRAINING",
                    feedback=self.training_feedback,
                    phase='training',
                )

            elif self.mode == 'test_only':
                self._screen2()
                self._show_fixation()
                self._set_phase('test')
                self.perf = self._empty_perf()
                self.task_clock.reset()
                self.run_block(
                    self.load_trials('Main'),
                    block_name="TEST",
                    feedback=False,
                    phase='test',
                )

            else:
                raise ValueError(f"[SART] Mode non reconnu : '{self.mode}'")

            self.logger.ok("[SART] Tâche terminée ✓")

        except (KeyboardInterrupt, SystemExit):
            self.logger.warn("[SART] Interruption détectée.")
            aborted = True

        except Exception as e:
            self.logger.warn(f"[SART] Erreur : {e}")
            traceback.print_exc()
            aborted = True

        finally:
            filepath = self.save_data()
            self._print_perf("FINAL")

            if self.doqc:
                qc_timing = self.test_timing or self.training_timing
                qc_label  = 'test' if self.test_timing else 'training'
                from tasks.qc.qc_sart import SARTTimingQC
                SARTTimingQC(
                    timing_log=qc_timing, frame_rate=self.frame_rate,
                    frame_dur_s=self.frame_dur_s,
                    digit_n_frames=self.digit_n_frames,
                    mask_n_frames=self.mask_n_frames,
                    participant_id=self.nom, session=self.session,
                    data_dir=self.data_dir, logger=self.logger,
                ).run_qc()
                self.logger.log(f"[SART] QC '{qc_label}' ({len(qc_timing)} essais)")

            if not aborted:
                self._show_screen("Fin de la tâche.\n\nMerci pour votre participation.")

        return filepath