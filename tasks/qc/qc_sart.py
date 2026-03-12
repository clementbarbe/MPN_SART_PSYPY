"""
qc_sart.py — Contrôle Qualité Timing pour le SART McGill
=========================================================
Vérifie que les durées réelles de présentation (digit, mask, SOA)
correspondent aux spécifications du protocole Robertson et al. (1997).

Verdict basé sur le POURCENTAGE d'essais hors-tolérance,
pas sur un seul outlier.

Seuils :
    WARN  : >1.5 frame par essai (±25ms @60Hz)
    ERROR : >2.5 frames par essai (±42ms @60Hz)

Verdict :
    PASS : <2% d'essais > ERROR threshold
    WARN : 2-10% d'essais > ERROR threshold, ou >10% > WARN threshold
    FAIL : >10% d'essais > ERROR threshold
"""

import os
import csv
import json
import numpy as np
from datetime import datetime


class SARTTimingQC:
    """Contrôle qualité des durées de présentation SART."""

    EXPECTED_DIGIT_MS = 250.0
    EXPECTED_MASK_MS  = 900.0
    EXPECTED_TOTAL_MS = 1150.0

    # ── Verdict thresholds (pourcentages) ──
    FAIL_PCT_ERROR = 10.0    # >10% essais dépassant ERROR → FAIL
    WARN_PCT_ERROR =  2.0    # >2% essais dépassant ERROR → WARN
    WARN_PCT_WARN  = 10.0    # >10% essais dépassant WARN → WARN

    def __init__(self, timing_log, frame_rate, frame_dur_s,
                 digit_n_frames, mask_n_frames,
                 participant_id, session, data_dir, logger):
        self.timing_log     = timing_log or []
        self.frame_rate     = frame_rate
        self.frame_dur_s    = frame_dur_s
        self.frame_dur_ms   = frame_dur_s * 1000
        self.digit_n_frames = digit_n_frames
        self.mask_n_frames  = mask_n_frames
        self.participant_id = participant_id
        self.session        = session
        self.data_dir       = data_dir
        self.logger         = logger

        # Seuils par essai (dynamiques)
        self.WARN_THRESHOLD_MS  = self.frame_dur_ms * 1.5
        self.ERROR_THRESHOLD_MS = self.frame_dur_ms * 2.5

        self.qc_dir = os.path.join(data_dir, 'qc')
        os.makedirs(self.qc_dir, exist_ok=True)

    # =================================================================
    #  ANALYSE
    # =================================================================
    def _compute_stats(self, values, label):
        if not values:
            return {
                'label': label, 'n': 0,
                'mean': 0, 'sd': 0, 'min': 0, 'max': 0,
                'abs_max': 0, 'median': 0,
                'n_warn': 0, 'n_error': 0, 'pct_warn': 0, 'pct_error': 0,
            }
        arr = np.array(values)
        abs_arr = np.abs(arr)
        n = len(arr)
        return {
            'label':     label,
            'n':         n,
            'mean':      round(float(np.mean(arr)), 3),
            'sd':        round(float(np.std(arr, ddof=1)) if n > 1 else 0.0, 3),
            'min':       round(float(np.min(arr)), 3),
            'max':       round(float(np.max(arr)), 3),
            'abs_max':   round(float(np.max(abs_arr)), 3),
            'median':    round(float(np.median(arr)), 3),
            'n_warn':    int(np.sum(abs_arr > self.WARN_THRESHOLD_MS)),
            'n_error':   int(np.sum(abs_arr > self.ERROR_THRESHOLD_MS)),
            'pct_warn':  round(float(np.sum(abs_arr > self.WARN_THRESHOLD_MS)) / n * 100, 1),
            'pct_error': round(float(np.sum(abs_arr > self.ERROR_THRESHOLD_MS)) / n * 100, 1),
        }

    def _compute_verdict(self, stats_dict):
        """
        Verdict basé sur le POURCENTAGE d'essais problématiques,
        pas sur un seul outlier.

        Examine digit, mask et total errors.
        """
        # Collecter les pct_error et pct_warn de chaque métrique
        max_pct_error = 0.0
        max_pct_warn  = 0.0

        for key in ['digit_error', 'mask_error', 'total_error']:
            s = stats_dict.get(key, {})
            if s.get('n', 0) == 0:
                continue
            max_pct_error = max(max_pct_error, s.get('pct_error', 0))
            max_pct_warn  = max(max_pct_warn,  s.get('pct_warn', 0))

        # SOA : seuils plus tolérants (inclut variabilité inter-trial)
        soa = stats_dict.get('soa_error', {})
        if soa.get('n', 0) > 0:
            soa_pct_error = soa.get('pct_error', 0)
            # SOA : on utilise la moitié du poids (outliers attendus aux transitions)
            max_pct_error = max(max_pct_error, soa_pct_error * 0.5)

        # Verdict
        if max_pct_error >= self.FAIL_PCT_ERROR:
            return 'FAIL'
        elif max_pct_error >= self.WARN_PCT_ERROR or max_pct_warn >= self.WARN_PCT_WARN:
            return 'WARN'
        else:
            return 'PASS'

    def analyze(self):
        if not self.timing_log:
            return None

        digit_errors = [t['digit_error_ms'] for t in self.timing_log]
        mask_errors  = [t['mask_error_ms'] for t in self.timing_log]
        total_errors = [t['total_error_ms'] for t in self.timing_log]

        digit_actual = [t['actual_digit_ms'] for t in self.timing_log]
        mask_actual  = [t['actual_mask_ms'] for t in self.timing_log]
        total_actual = [t['actual_total_ms'] for t in self.timing_log]

        soa_errors = []
        for i in range(1, len(self.timing_log)):
            soa_ms = (self.timing_log[i]['digit_onset']
                       - self.timing_log[i-1]['digit_onset']) * 1000
            soa_errors.append(soa_ms - self.EXPECTED_TOTAL_MS)

        stats = {
            'digit_error':  self._compute_stats(digit_errors, 'Digit Error (ms)'),
            'mask_error':   self._compute_stats(mask_errors, 'Mask Error (ms)'),
            'total_error':  self._compute_stats(total_errors, 'Total Error (ms)'),
            'soa_error':    self._compute_stats(soa_errors, 'SOA Error (ms)'),
            'digit_actual': self._compute_stats(digit_actual, 'Digit Actual (ms)'),
            'mask_actual':  self._compute_stats(mask_actual, 'Mask Actual (ms)'),
            'total_actual': self._compute_stats(total_actual, 'Total Actual (ms)'),
        }

        verdict = self._compute_verdict(stats)

        return {
            'participant':        self.participant_id,
            'session':            self.session,
            'frame_rate':         round(self.frame_rate, 2),
            'frame_dur_ms':       round(self.frame_dur_ms, 3),
            'digit_n_frames':     self.digit_n_frames,
            'mask_n_frames':      self.mask_n_frames,
            'n_trials':           len(self.timing_log),
            'warn_threshold_ms':  round(self.WARN_THRESHOLD_MS, 2),
            'error_threshold_ms': round(self.ERROR_THRESHOLD_MS, 2),
            'verdict_criteria': {
                'FAIL': f'>{self.FAIL_PCT_ERROR}% trials > error threshold',
                'WARN': f'>{self.WARN_PCT_ERROR}% trials > error threshold '
                        f'OR >{self.WARN_PCT_WARN}% > warn threshold',
                'PASS': 'otherwise',
            },
            'stats':   stats,
            'verdict': verdict,
        }

    # =================================================================
    #  RAPPORT CONSOLE
    # =================================================================
    def print_report(self, analysis):
        if not analysis:
            print("[QC] Aucune donnée de timing à analyser.")
            return

        v = analysis['verdict']
        v_icon = {'PASS': '✅', 'WARN': '⚠️', 'FAIL': '❌'}.get(v, '?')

        print(f"\n{'='*65}")
        print(f"  SART TIMING QUALITY CONTROL — {v_icon} {v}")
        print(f"{'='*65}")
        print(f"  Participant   : {analysis['participant']}")
        print(f"  Session       : {analysis['session']}")
        print(f"  Trials        : {analysis['n_trials']}")
        print(f"  Frame rate    : {analysis['frame_rate']:.2f} Hz")
        print(f"  Frame duration: {analysis['frame_dur_ms']:.2f} ms")
        print(f"  Digit frames  : {analysis['digit_n_frames']} "
              f"→ {analysis['digit_n_frames'] * analysis['frame_dur_ms']:.1f} ms "
              f"(target: 250 ms)")
        print(f"  Mask frames   : {analysis['mask_n_frames']} "
              f"→ {analysis['mask_n_frames'] * analysis['frame_dur_ms']:.1f} ms "
              f"(target: 900 ms)")
        print(f"  Warn threshold: ±{analysis['warn_threshold_ms']:.1f} ms "
              f"(1.5 frames)")
        print(f"  Error threshold: ±{analysis['error_threshold_ms']:.1f} ms "
              f"(2.5 frames)")
        print(f"  {'─'*61}")
        print(f"  Verdict criteria:")
        print(f"    PASS : <{self.WARN_PCT_ERROR}% errors AND <{self.WARN_PCT_WARN}% warnings")
        print(f"    WARN : {self.WARN_PCT_ERROR}-{self.FAIL_PCT_ERROR}% errors "
              f"OR >{self.WARN_PCT_WARN}% warnings")
        print(f"    FAIL : >{self.FAIL_PCT_ERROR}% errors")
        print(f"  {'─'*61}")

        for key in ['digit_error', 'mask_error', 'total_error', 'soa_error']:
            s = analysis['stats'][key]
            if s['n'] == 0:
                continue

            if s['pct_error'] >= self.FAIL_PCT_ERROR:
                flag = ' ❌'
            elif (s['pct_error'] >= self.WARN_PCT_ERROR or
                  s['pct_warn'] >= self.WARN_PCT_WARN):
                flag = ' ⚠️'
            else:
                flag = ' ✅'

            print(f"\n  {s['label']}{flag}")
            print(f"    Mean  : {s['mean']:+.2f} ms")
            print(f"    SD    : {s['sd']:.2f} ms")
            print(f"    Median: {s['median']:+.2f} ms")
            print(f"    Range : [{s['min']:+.2f}, {s['max']:+.2f}] ms")
            print(f"    |Max| : {s['abs_max']:.2f} ms")
            print(f"    >Warn : {s['n_warn']}/{s['n']} ({s['pct_warn']:.1f}%)")
            print(f"    >Error: {s['n_error']}/{s['n']} ({s['pct_error']:.1f}%)")

            # Contextualiser les outliers
            if s['n_error'] > 0 and s['pct_error'] < self.FAIL_PCT_ERROR:
                print(f"    ℹ️  {s['n_error']} outlier(s) isolé(s) — "
                      f"probablement un frame drop OS/GC (non systémique)")

        print(f"\n  {'─'*61}")
        ideal_digit = analysis['digit_n_frames'] * analysis['frame_dur_ms']
        ideal_mask  = analysis['mask_n_frames'] * analysis['frame_dur_ms']
        print(f"  ℹ️  Frame-budget digit : {analysis['digit_n_frames']} × "
              f"{analysis['frame_dur_ms']:.2f} = {ideal_digit:.1f} ms "
              f"(cible: 250 ms, écart: {ideal_digit - 250:.1f} ms)")
        print(f"  ℹ️  Frame-budget mask  : {analysis['mask_n_frames']} × "
              f"{analysis['frame_dur_ms']:.2f} = {ideal_mask:.1f} ms "
              f"(cible: 900 ms, écart: {ideal_mask - 900:.1f} ms)")

        if abs(ideal_digit - 250) > analysis['frame_dur_ms']:
            print(f"  ⚠️  Le taux ({analysis['frame_rate']:.1f} Hz) "
                  f"ne permet pas un digit de 250 ms exact !")

        if abs(ideal_mask - 900) > analysis['frame_dur_ms']:
            print(f"  ⚠️  Le taux ({analysis['frame_rate']:.1f} Hz) "
                  f"ne permet pas un mask de 900 ms exact !")

        print(f"\n{'='*65}")
        print(f"  VERDICT : {v_icon} {v}")
        if v == 'PASS':
            print("  Toutes les durées sont dans les limites acceptables.")
        elif v == 'WARN':
            print("  Quelques essais dévient. Vérifiez le CSV pour les outliers.")
            print("  Les données restent vraisemblablement utilisables.")
        else:
            print("  ❌ PROBLÈMES DE TIMING SYSTÉMIQUES DÉTECTÉS !")
            print(f"  Plus de {self.FAIL_PCT_ERROR}% des essais dépassent "
                  f"le seuil d'erreur ({analysis['error_threshold_ms']:.0f} ms).")
            print("  → Vérifiez la charge CPU, le mode GPU, et le V-Sync.")
        print(f"{'='*65}\n")

    # =================================================================
    #  SOA CHECK
    # =================================================================
    def check_soa_regularity(self):
        if len(self.timing_log) < 2:
            return

        soas = []
        for i in range(1, len(self.timing_log)):
            soa_ms = (self.timing_log[i]['digit_onset']
                       - self.timing_log[i-1]['digit_onset']) * 1000
            soas.append(soa_ms)

        soa_arr = np.array(soas)
        mean_soa = float(np.mean(soa_arr))
        max_soa  = float(np.max(soa_arr))
        min_soa  = float(np.min(soa_arr))

        print(f"\n  SOA Inter-Trial Check (onset → onset)")
        print(f"    Expected : {self.EXPECTED_TOTAL_MS:.0f} ms")
        print(f"    Mean     : {mean_soa:.1f} ms")
        print(f"    Range    : [{min_soa:.1f}, {max_soa:.1f}] ms")

        if mean_soa > self.EXPECTED_TOTAL_MS + 50:
            excess = mean_soa - self.EXPECTED_TOTAL_MS
            self.logger.warn(
                f"[QC] ⚠️  SOA moyen = {mean_soa:.0f} ms — "
                f"excès de {excess:.0f} ms par rapport au protocole."
            )
            print(f"    ⚠️  SOA moyen dépasse le protocole de {excess:.0f} ms !")
        elif mean_soa < self.EXPECTED_TOTAL_MS - 50:
            self.logger.warn(
                f"[QC] ⚠️  SOA moyen = {mean_soa:.0f} ms — "
                f"inférieur au protocole."
            )
        else:
            print(f"    ✅ SOA conforme au protocole (±50 ms)")

        # Identifier les outliers SOA
        soa_errors = np.abs(soa_arr - self.EXPECTED_TOTAL_MS)
        n_outliers = int(np.sum(soa_errors > self.ERROR_THRESHOLD_MS * 2))
        if n_outliers > 0 and n_outliers <= 3:
            print(f"    ℹ️  {n_outliers} SOA outlier(s) — probablement frame drop(s) isolé(s)")

    # =================================================================
    #  STRUCTURE CHECK
    # =================================================================
    def check_trial_structure(self):
        if not self.timing_log:
            return

        conditions = [t['condition'] for t in self.timing_log]
        n_go   = conditions.count('go')
        n_nogo = conditions.count('nogo')
        total  = len(conditions)

        print(f"\n  Trial Structure Check")
        print(f"    Total   : {total}")
        print(f"    GO      : {n_go}")
        print(f"    NO-GO   : {n_nogo}")
        print(f"    Ratio   : 1 NO-GO / {n_go/max(n_nogo,1):.1f} GO")

        if total == 225 and n_go == 200 and n_nogo == 25:
            print(f"    ✅ Conforme (225 = 200 GO + 25 NO-GO)")
        elif total == 20 and n_go == 18 and n_nogo == 2:
            print(f"    ✅ Conforme Training (20 = 18 GO + 2 NO-GO)")
        else:
            ratio_ok = abs(n_go / max(n_nogo, 1) - 8.0) < 1