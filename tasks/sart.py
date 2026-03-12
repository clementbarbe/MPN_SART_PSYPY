"""
sart.py — Sustained Attention to Response Task (McGill Protocol)
================================================================
Protocole fidèle à Robertson et al. (1997) et à l'annexe McGill :
    • 20 essais d'entraînement (18 GO + 2 NO-GO), ordre fixe, avec feedback
    • 225 essais test (200 GO + 25 NO-GO), ordre fixe (Excel), sans feedback
    • Digit 250 ms → Mask 900 ms → digit suivant immédiat (SOA = 1150 ms, PAS d'ISI)
    • 4 tailles de police (tiny, small, medium, large)
    • 3 écrans d'instructions (FR) avec visuel du masque
    • Métriques : d', criterion c, beta, RTCV

Entrée  : SART_trials_McGill.xlsx
Sortie  : McGill_SART_Raw_Data_*.xlsx  +  data/sart/qc/SART_TimingQC_*.csv
"""

import gc
import os
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import norm

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
}

MODE_ALIASES = {
    'full':          'full',
    'training':      'training_only',
    'training_only': 'training_only',
    'train':         'training_only',
    'test':          'test_only',
    'test_only':     'test_only',
}

SIZE_MAP = {
    'tiny':   {'pct': '2pct',  'height': 0.04},
    'small':  {'pct': '4pct',  'height': 0.06},
    'medium': {'pct': '10pct', 'height': 0.10},
    'large':  {'pct': '16pct', 'height': 0.14},
}

KEY_CODE_MAP = {'space': 57}
TOTAL_TRIAL_MS = 1150

# ── Séquence fixe d'entraînement McGill (20 essais : 18 GO + 2 NO-GO) ──
TRAINING_SEQUENCE = [
    (1, 'medium'), (5, 'small'),  (7, 'large'), (9, 'tiny'),
    (2, 'medium'), (6, 'small'),  (8, 'large'), (4, 'tiny'),
    (1, 'small'),  (3, 'medium'),
    (7, 'large'),  (9, 'tiny'),   (5, 'medium'), (2, 'small'),
    (8, 'large'),  (6, 'tiny'),   (3, 'small'),
    (4, 'medium'), (1, 'large'),  (9, 'small'),
]


class sart(BaseTask):
    """Sustained Attention to Response Task — McGill Protocol."""

    DIGITS           = list(range(1, 10))
    DIGIT_DURATION_S = 0.250
    MASK_DURATION_S  = 0.900
    MASK_RADIUS      = 0.08

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
            enregistrer=True
        )

        # ── Normalisation du mode ──
        raw_mode = cfg['mode'].lower().strip()
        self.mode = MODE_ALIASES.get(raw_mode)
        if self.mode is None:
            self.logger.warn(
                f"[SART] Mode '{raw_mode}' non reconnu — fallback 'full'. "
                f"Valides : {list(MODE_ALIASES.keys())}"
            )
            self.mode = 'full'

        self.target_digit = int(cfg['target_digit'])
        self.response_key = cfg['response_key']
        self.trial_file   = cfg['trial_file']
        self.feedback_dur = float(cfg['feedback_duration'])

        # ── Données séparées par phase ──
        self.training_data    = []
        self.training_timing  = []
        self.test_data        = []
        self.test_timing      = []

        # Pointeurs actifs (changent selon la phase)
        self.trial_data       = self.training_data
        self.timing_log       = self.training_timing
        self.global_records   = []     # alias pour BaseTask._emergency_save

        self.perf = self._empty_perf()

        self._measure_frame_rate()
        self._setup_stimuli()

        self.logger.log(
            f"[SART] Init | mode={self.mode} (raw='{raw_mode}') | "
            f"target={self.target_digit} | "
            f"fps={self.frame_rate:.1f} | frame={self.frame_dur_s*1000:.2f}ms"
        )

    # =================================================================
    # HELPERS
    # =================================================================
    @staticmethod
    def _empty_perf():
        return {
            'go_correct': 0, 'go_omission': 0,
            'nogo_correct': 0, 'nogo_commission': 0,
            'go_rts': [],
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
            raise ValueError(f"Phase inconnue : {phase}")

        # Mise à jour de l'alias pour emergency save
        self.global_records = self.trial_data

        self.logger.log(f"[SART] Phase → {phase}")

    @staticmethod
    def _size_text_to_info(size_text):
        key = str(size_text).strip().lower()
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
        self.frame_rate    = measured if measured else 60.0
        self.frame_dur_s   = 1.0 / self.frame_rate
        self.digit_n_frames = max(1, round(self.DIGIT_DURATION_S / self.frame_dur_s))
        self.mask_n_frames  = max(1, round(self.MASK_DURATION_S / self.frame_dur_s))
        self.logger.log(
            f"[SART] Digit={self.digit_n_frames}f "
            f"({self.digit_n_frames * self.frame_dur_s * 1000:.1f}ms) | "
            f"Mask={self.mask_n_frames}f "
            f"({self.mask_n_frames * self.frame_dur_s * 1000:.1f}ms)"
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
            self.mask_circle.pos = (0.0, -0.10)
            self.mask_cross_a.start = (-self.MASK_RADIUS * 0.70,
                                       -self.MASK_RADIUS * 0.70 - 0.10)
            self.mask_cross_a.end   = ( self.MASK_RADIUS * 0.70,
                                        self.MASK_RADIUS * 0.70 - 0.10)
            self.mask_cross_b.start = (-self.MASK_RADIUS * 0.70,
                                        self.MASK_RADIUS * 0.70 - 0.10)
            self.mask_cross_b.end   = ( self.MASK_RADIUS * 0.70,
                                       -self.MASK_RADIUS * 0.70 - 0.10)
            self._draw_mask()
            # Reset
            self.mask_circle.pos = (0.0, 0.0)
            arm = self.MASK_RADIUS * 0.70
            self.mask_cross_a.start = (-arm, -arm)
            self.mask_cross_a.end   = ( arm,  arm)
            self.mask_cross_b.start = (-arm,  arm)
            self.mask_cross_b.end   = ( arm, -arm)

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
        self._show_screen(txt, show_mask=True)

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
    # TRIAL LOADING
    # =================================================================
    def load_trials_from_excel(self):
        filepath = self.trial_file
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"[SART] Fichier introuvable : {filepath}")

        df = pd.read_excel(filepath)
        df.columns = [c.strip().lower() for c in df.columns]

        trials = []
        for _, row in df.iterrows():
            digit    = int(row['digit'])
            is_nogo  = int(row['isnogo'])
            size_txt = str(row['size']).strip()

            pct_label, norm_height = self._size_text_to_info(size_txt)

            if norm_height not in [v['height'] for v in SIZE_MAP.values()]:
                self.logger.warn(
                    f"[SART] Taille inattendue '{size_txt}' → forçage 'medium'."
                )
                pct_label   = SIZE_MAP['medium']['pct']
                norm_height = SIZE_MAP['medium']['height']

            condition        = 'nogo' if is_nogo == 1 else 'go'
            correct_response = 0 if is_nogo == 1 else 1

            trial_type_raw = str(row.get('trialtype', condition)).strip()
            if trial_type_raw.upper() == 'NO-GO':
                trial_type_out = 'No-Go'
            elif trial_type_raw.upper() == 'GO':
                trial_type_out = 'Go'
            else:
                trial_type_out = 'Go' if condition == 'go' else 'No-Go'

            trials.append({
                'trialnum':         int(row['trial']),
                'digit':            digit,
                'font_size_norm':   norm_height,
                'font_size_label':  pct_label,
                'condition':        condition,
                'correct_response': correct_response,
                'trial_type_out':   trial_type_out,
            })

        n_go   = sum(1 for t in trials if t['condition'] == 'go')
        n_nogo = sum(1 for t in trials if t['condition'] == 'nogo')
        self.logger.log(
            f"[SART] Excel chargé : {len(trials)} essais ({n_go} GO, {n_nogo} NO-GO)"
        )
        if n_go != 200 or n_nogo != 25:
            self.logger.warn(
                f"[SART] ⚠️ Attendu: 200 GO + 25 NO-GO, "
                f"Trouvé: {n_go} GO + {n_nogo} NO-GO"
            )
        return trials

    def build_training_trials(self):
        trials = []
        for i, (digit, size_key) in enumerate(TRAINING_SEQUENCE, start=1):
            condition = 'nogo' if digit == self.target_digit else 'go'
            trials.append({
                'trialnum':         i,
                'digit':            digit,
                'font_size_norm':   SIZE_MAP[size_key]['height'],
                'font_size_label':  SIZE_MAP[size_key]['pct'],
                'condition':        condition,
                'correct_response': 1 if condition == 'go' else 0,
                'trial_type_out':   'Go' if condition == 'go' else 'No-Go',
            })

        n_go   = sum(1 for t in trials if t['condition'] == 'go')
        n_nogo = sum(1 for t in trials if t['condition'] == 'nogo')
        self.logger.log(f"[SART] Training: {len(trials)} essais ({n_go} GO, {n_nogo} NO-GO)")
        assert len(trials) == 20
        assert n_nogo == 2
        return trials

    # =================================================================
    # CORE TRIAL — FIX RT : utiliser tDown (absolu) au lieu de rt (relatif)
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
        responded_in = None

        self.flush_keyboard()

        # ── PHASE 1 — DIGIT (250 ms) ──────────────────────────
        self._draw_digit(digit, font_size)
        t_digit_onset = self.win.flip()

        for _ in range(self.digit_n_frames - 1):
            if not responded:
                keys = self.get_keys([self.response_key])
                if keys:
                    responded    = True
                    response_key = keys[0].name
                    response_rt  = keys[0].tDown - t_digit_onset
                    responded_in = 'digit'
            self._draw_digit(digit, font_size)
            self.win.flip()

        # ── PHASE 2 — MASK (900 ms) ───────────────────────────
        self._draw_mask()
        t_mask_onset = self.win.flip()

        for frame_i in range(self.mask_n_frames - 1):
            if not responded:
                keys = self.get_keys([self.response_key])
                if keys:
                    responded    = True
                    response_key = keys[0].name
                    response_rt  = keys[0].tDown - t_digit_onset
                    responded_in = 'mask'
            self._draw_mask()
            # ╔══════════════════════════════════════════════════╗
            # ║  Capturer le timestamp du DERNIER flip du mask   ║
            # ║  pour mesurer la vraie durée du masque           ║
            # ╚══════════════════════════════════════════════════╝
            t_last_flip = self.win.flip()

        # Le dernier flip du mask = moment réel où le masque disparaît
        # (le prochain trial dessinera par-dessus au flip suivant)
        t_mask_offset = t_last_flip

        # ── QC TIMING — MESURES RÉELLES ────────────────────────
        actual_digit_ms = (t_mask_onset - t_digit_onset) * 1000
        actual_mask_ms  = (t_mask_offset - t_mask_onset) * 1000
        actual_total_ms = (t_mask_offset - t_digit_onset) * 1000

        timing_record = {
            'trial':            trial_index,
            'phase':            phase,
            'digit':            digit,
            'condition':        condition,
            'digit_onset':      round(t_digit_onset, 6),
            'mask_onset':       round(t_mask_onset, 6),
            'mask_offset':      round(t_mask_offset, 6),
            'actual_digit_ms':  round(actual_digit_ms, 2),
            'actual_mask_ms':   round(actual_mask_ms, 2),
            'actual_total_ms':  round(actual_total_ms, 2),
            'expected_digit_ms': 250.0,
            'expected_mask_ms':  900.0,
            'expected_total_ms': 1150.0,
            'digit_error_ms':   round(actual_digit_ms - 250.0, 2),
            'mask_error_ms':    round(actual_mask_ms - 900.0, 2),
            'total_error_ms':   round(actual_total_ms - 1150.0, 2),
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

        # ── CLASSIFICATION ──────────────────────────────────────
        if is_nogo:
            if responded:
                accuracy = 'Commission Error'
                corr = 0
                self.perf['nogo_commission'] += 1
            else:
                accuracy = 'No-Go Success'
                corr = 1
                self.perf['nogo_correct'] += 1
        else:
            if responded:
                accuracy = 'Go Success'
                corr = 1
                self.perf['go_correct'] += 1
                self.perf['go_rts'].append(response_rt)
            else:
                accuracy = 'Omission'
                corr = 0
                self.perf['go_omission'] += 1

        if responded:
            response_code = self._key_to_code(response_key)
            rt_ms         = round(response_rt * 1000)
            latency       = rt_ms
            latency_type  = 1 if responded_in == 'digit' else 3
        else:
            response_code = 0
            rt_ms         = ''
            latency       = TOTAL_TRIAL_MS
            latency_type  = 0

        # ── Validation RT ──────────────────────────────────────
        if responded and (rt_ms < 0 or rt_ms > TOTAL_TRIAL_MS + 100):
            self.logger.warn(
                f"[SART] ⚠️ RT aberrant trial {trial_index}: {rt_ms}ms "
                f"(tDown={keys[0].tDown:.6f}, onset={t_digit_onset:.6f})"
            )

        # ── FEEDBACK (training) ────────────────────────────────
        if feedback:
            if corr == 1:
                self.fb_symbol.text, self.fb_symbol.color = '✓', 'green'
                if accuracy == 'Go Success':
                    self.fb_msg.text = f"Correct ! RT : {rt_ms} ms"
                else:
                    self.fb_msg.text = "Correct ! Bonne inhibition ✓"
            else:
                self.fb_symbol.text, self.fb_symbol.color = '✗', 'red'
                if accuracy == 'Commission Error':
                    self.fb_msg.text = (
                        f"Erreur — ne PAS appuyer pour le {self.target_digit}"
                    )
                else:
                    self.fb_msg.text = "Erreur — appuyez pour les autres chiffres"

            self.fb_symbol.draw()
            self.fb_msg.draw()
            self.win.flip()
            core.wait(self.feedback_dur)

        # ── ENREGISTREMENT ────────────────────────────────────
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
            'actual_digit_ms':         timing_record['actual_digit_ms'],
            'actual_mask_ms':          timing_record['actual_mask_ms'],
            'actual_total_ms':         timing_record['actual_total_ms'],
            'digit_error_ms':          timing_record['digit_error_ms'],
            'mask_error_ms':           timing_record['mask_error_ms'],
            'total_error_ms':          timing_record['total_error_ms'],
            'countAnticipatory':       0,
            'correctSuppressions':     0,
            'incorrectSuppressions':   0,
            'countNoGo':               0,
            'countGo':                 0,
            'countValidGo':            0,
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
                phase=phase
            )
        self._print_performance(block_name)

    def compute_metrics(self):
        p = self.perf
        total_go   = p['go_correct'] + p['go_omission']
        total_nogo = p['nogo_correct'] + p['nogo_commission']

        go_acc   = (p['go_correct'] / total_go * 100) if total_go else 0
        nogo_acc = (p['nogo_correct'] / total_nogo * 100) if total_nogo else 0

        if p['go_rts']:
            rts       = np.array(p['go_rts']) * 1000
            mean_rt   = float(np.mean(rts))
            sd_rt     = float(np.std(rts, ddof=1)) if len(rts) > 1 else 0.0
            rtcv      = sd_rt / mean_rt if mean_rt > 0 else 0.0
            median_rt = float(np.median(rts))
        else:
            mean_rt = sd_rt = rtcv = median_rt = 0.0

        hit_rate = np.clip(p['go_correct'] / total_go, 0.01, 0.99) if total_go else 0.5
        fa_rate  = np.clip(p['nogo_commission'] / total_nogo, 0.01, 0.99) if total_nogo else 0.5

        z_hit = float(norm.ppf(hit_rate))
        z_fa  = float(norm.ppf(fa_rate))

        d_prime     = z_hit - z_fa if (total_go and total_nogo) else 0.0
        criterion_c = -0.5 * (z_hit + z_fa) if (total_go and total_nogo) else 0.0
        beta        = float(np.exp(criterion_c * d_prime)) if d_prime != 0 else 1.0

        return {
            'total_go': total_go, 'total_nogo': total_nogo,
            'go_correct': p['go_correct'], 'go_omission': p['go_omission'],
            'nogo_correct': p['nogo_correct'], 'nogo_commission': p['nogo_commission'],
            'go_accuracy_pct': round(go_acc, 1), 'nogo_accuracy_pct': round(nogo_acc, 1),
            'mean_rt_ms': round(mean_rt, 1), 'median_rt_ms': round(median_rt, 1),
            'sd_rt_ms': round(sd_rt, 1), 'rtcv': round(rtcv, 4),
            'd_prime': round(d_prime, 3), 'criterion_c': round(criterion_c, 3),
            'beta': round(beta, 3),
        }

    def _print_performance(self, block_name=""):
        m = self.compute_metrics()
        print(
            f"\n{'─'*55}\n  SART Performance — {block_name}\n{'─'*55}\n"
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
            f"  Criterion c  : {m['criterion_c']:+.3f}\n"
            f"  Beta (β)     : {m['beta']:.3f}\n"
            f"{'─'*55}"
        )

    # =================================================================
    # SAVE — Excel obligatoire, training & test sur 2 feuilles
    # =================================================================
    def save_data(self, **kwargs):
        """Sauvegarde en Excel avec feuilles séparées Training / Test."""
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
            'actual_digit_ms', 'actual_mask_ms', 'digit_error_ms', 'mask_error_ms',
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
                    df_train = _to_df(self.training_data)
                    df_train.to_excel(writer, sheet_name='Training', index=False)
                    self.logger.log(
                        f"[SART] Training: {len(self.training_data)} essais → feuille 'Training'"
                    )

                if has_test:
                    df_test = _to_df(self.test_data)
                    df_test.to_excel(writer, sheet_name='Test', index=False)
                    self.logger.log(
                        f"[SART] Test: {len(self.test_data)} essais → feuille 'Test'"
                    )

                # Feuille combinée pour compatibilité
                all_data = self.training_data + self.test_data
                if all_data:
                    df_all = _to_df(all_data)
                    df_all.to_excel(writer, sheet_name='All_Trials', index=False)

            self.logger.ok(f"[SART] Données (Excel McGill) → {filepath}")

        except Exception as e:
            self.logger.warn(f"[SART] Erreur sauvegarde Excel : {e}")
            # Fallback : CSV d'urgence
            csv_path = filepath.replace('.xlsx', '_EMERGENCY.csv')
            all_data = self.training_data + self.test_data
            pd.DataFrame(all_data).to_csv(csv_path, index=False)
            self.logger.warn(f"[SART] Fallback CSV → {csv_path}")
            filepath = csv_path

        return filepath

    # =================================================================
    # MAIN RUN
    # =================================================================
    def run(self):
        filepath = None
        aborted  = False

        try:
            self.show_instructions_screen1()

            do_training = self.mode in ('full', 'training_only')
            do_test     = self.mode in ('full', 'test_only')

            self.logger.log(
                f"[SART] run() | mode={self.mode} | "
                f"do_training={do_training} | do_test={do_test}"
            )

            # ── TRAINING ──────────────────────────────────────
            if do_training:
                self.show_instructions_screen2()

                # Basculer vers stockage training
                self._set_phase('training')
                self.perf = self._empty_perf()
                self.task_clock.reset()

                training_trials = self.build_training_trials()
                self.run_block(
                    training_trials, block_name="TRAINING",
                    feedback=True, phase='training'
                )

            # ── TEST ──────────────────────────────────────────
            if do_test:
                if do_training:
                    self.show_instructions_screen3()
                else:
                    self.show_instructions_screen2()

                # Basculer vers stockage test + reset perf
                self._set_phase('test')
                self.perf = self._empty_perf()
                self.task_clock.reset()

                test_trials = self.load_trials_from_excel()
                self.run_block(
                    test_trials, block_name="TEST",
                    feedback=False, phase='test'
                )

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

            # ── QC : uniquement sur les données TEST ──────────
            qc_timing = self.test_timing if self.test_timing else self.training_timing
            qc_label  = 'test' if self.test_timing else 'training'

            try:
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
            except Exception as e:
                self.logger.warn(f"[SART] QC timing échoué : {e}")
                import traceback
                traceback.print_exc()

            if not aborted:
                self._show_screen(
                    "Fin de la tâche.\n\nMerci pour votre participation.",
                    show_mask=False
                )

        return filepath