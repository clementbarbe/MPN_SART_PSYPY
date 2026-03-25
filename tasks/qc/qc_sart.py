"""
qc_sart.py — Contrôle Qualité Timing SART McGill
=================================================
Vérifie la stabilité technique et temporelle de la tâche.

Verdict basé sur le pourcentage d'essais hors-tolérance :
    PASS : <2%  d'essais > seuil erreur
    WARN : 2-10% d'essais > seuil erreur OU >10% > seuil warn
    FAIL : >10% d'essais > seuil erreur

Usage :
    qc = SARTTimingQC(timing_log=..., frame_rate=..., frame_dur_s=...,
                      digit_n_frames=..., mask_n_frames=...,
                      participant_id=..., session=...,
                      data_dir=..., logger=...)
    qc.run_qc()
"""

import os
import csv
import numpy as np
from datetime import datetime


class SARTTimingQC:

    EXPECTED_DIGIT_MS = 250.0
    EXPECTED_MASK_MS  = 900.0
    EXPECTED_TOTAL_MS = 1150.0

    FAIL_PCT = 10.0   # >10% essais > seuil erreur → FAIL
    WARN_PCT =  2.0   # >2%  essais > seuil erreur → WARN
    WARN_PCT_SOFT = 10.0  # >10% essais > seuil warn → WARN

    def __init__(self, timing_log, frame_rate, frame_dur_s,
                 digit_n_frames, mask_n_frames,
                 participant_id, session, data_dir, logger):

        self.timing_log     = timing_log or []
        self.frame_rate     = frame_rate
        self.frame_dur_ms   = frame_dur_s * 1000
        self.digit_n_frames = digit_n_frames
        self.mask_n_frames  = mask_n_frames
        self.participant_id = participant_id
        self.session        = session
        self.data_dir       = data_dir
        self.logger         = logger

        # Seuils dynamiques (1.5 et 2.5 frames)
        self.warn_ms  = self.frame_dur_ms * 1.5
        self.error_ms = self.frame_dur_ms * 2.5

        self.qc_dir = os.path.join(data_dir, 'qc')
        os.makedirs(self.qc_dir, exist_ok=True)

    # =================================================================
    # POINT D'ENTRÉE UNIQUE
    # =================================================================
    def run_qc(self):
        """Lance l'analyse complète, affiche le rapport et sauvegarde le CSV."""
        if not self.timing_log:
            self.logger.warn("[QC] Aucune donnée de timing — QC ignoré.")
            return

        results = self._analyze()
        self._print_report(results)
        self._save_csv(results)
        return results

    # =================================================================
    # ANALYSE
    # =================================================================
    def _analyze(self):
        digit_errors = [t['digit_error_ms'] for t in self.timing_log]
        mask_errors  = [t['mask_error_ms']  for t in self.timing_log]
        total_errors = [t['total_error_ms'] for t in self.timing_log]

        # SOA inter-essai (onset[i] - onset[i-1])
        soa_errors = [
            (self.timing_log[i]['digit_onset'] -
             self.timing_log[i-1]['digit_onset']) * 1000 - self.EXPECTED_TOTAL_MS
            for i in range(1, len(self.timing_log))
        ]

        stats = {
            'digit': self._stats(digit_errors, 'Digit  (cible 250 ms)'),
            'mask':  self._stats(mask_errors,  'Mask   (cible 900 ms)'),
            'total': self._stats(total_errors, 'Total  (cible 1150 ms)'),
            'soa':   self._stats(soa_errors,   'SOA inter-essai'),
        }

        verdict = self._verdict(stats)

        return {
            'participant':    self.participant_id,
            'session':        self.session,
            'n_trials':       len(self.timing_log),
            'frame_rate_hz':  round(self.frame_rate, 2),
            'frame_dur_ms':   round(self.frame_dur_ms, 3),
            'digit_n_frames': self.digit_n_frames,
            'mask_n_frames':  self.mask_n_frames,
            'warn_ms':        round(self.warn_ms, 2),
            'error_ms':       round(self.error_ms, 2),
            'stats':          stats,
            'verdict':        verdict,
        }

    def _stats(self, values, label):
        if not values:
            return {'label': label, 'n': 0}
        arr = np.array(values)
        abs_arr = np.abs(arr)
        n = len(arr)
        return {
            'label':     label,
            'n':         n,
            'mean':      round(float(np.mean(arr)),   3),
            'sd':        round(float(np.std(arr, ddof=1)) if n > 1 else 0.0, 3),
            'median':    round(float(np.median(arr)), 3),
            'min':       round(float(np.min(arr)),    3),
            'max':       round(float(np.max(arr)),    3),
            'abs_max':   round(float(np.max(abs_arr)),3),
            'n_warn':    int(np.sum(abs_arr > self.warn_ms)),
            'n_error':   int(np.sum(abs_arr > self.error_ms)),
            'pct_warn':  round(float(np.mean(abs_arr > self.warn_ms))  * 100, 1),
            'pct_error': round(float(np.mean(abs_arr > self.error_ms)) * 100, 1),
        }

    def _verdict(self, stats):
        max_pct_error = 0.0
        max_pct_warn  = 0.0
        for key in ('digit', 'mask', 'total'):
            s = stats[key]
            if s.get('n', 0) == 0:
                continue
            max_pct_error = max(max_pct_error, s['pct_error'])
            max_pct_warn  = max(max_pct_warn,  s['pct_warn'])

        if max_pct_error >= self.FAIL_PCT:
            return 'FAIL'
        if max_pct_error >= self.WARN_PCT or max_pct_warn >= self.WARN_PCT_SOFT:
            return 'WARN'
        return 'PASS'

    # =================================================================
    # RAPPORT CONSOLE
    # =================================================================
    def _print_report(self, r):
        icon = {'PASS': '✅', 'WARN': '⚠️', 'FAIL': '❌'}.get(r['verdict'], '?')

        print(f"\n{'='*60}")
        print(f"  SART TIMING QC — {icon} {r['verdict']}")
        print(f"{'='*60}")
        print(f"  Participant : {r['participant']}   Session : {r['session']}")
        print(f"  Essais      : {r['n_trials']}")
        print(f"  Frame rate  : {r['frame_rate_hz']:.2f} Hz  "
              f"({r['frame_dur_ms']:.2f} ms/frame)")
        print(f"  Budget digit: {r['digit_n_frames']} × {r['frame_dur_ms']:.2f} = "
              f"{r['digit_n_frames'] * r['frame_dur_ms']:.1f} ms  "
              f"(cible 250 ms, écart {r['digit_n_frames'] * r['frame_dur_ms'] - 250:.1f} ms)")
        print(f"  Budget mask : {r['mask_n_frames']} × {r['frame_dur_ms']:.2f} = "
              f"{r['mask_n_frames'] * r['frame_dur_ms']:.1f} ms  "
              f"(cible 900 ms, écart {r['mask_n_frames'] * r['frame_dur_ms'] - 900:.1f} ms)")
        print(f"  Seuil warn  : ±{r['warn_ms']:.1f} ms (1.5 frames)")
        print(f"  Seuil error : ±{r['error_ms']:.1f} ms (2.5 frames)")
        print(f"  {'─'*56}")

        for key in ('digit', 'mask', 'total', 'soa'):
            s = r['stats'][key]
            if s.get('n', 0) == 0:
                continue

            if s['pct_error'] >= self.FAIL_PCT:
                flag = '❌'
            elif s['pct_error'] >= self.WARN_PCT or s['pct_warn'] >= self.WARN_PCT_SOFT:
                flag = '⚠️'
            else:
                flag = '✅'

            print(f"\n  {flag}  {s['label']}")
            print(f"      Moyenne ± SD : {s['mean']:+.2f} ± {s['sd']:.2f} ms")
            print(f"      Médiane      : {s['median']:+.2f} ms")
            print(f"      Étendue      : [{s['min']:+.2f}, {s['max']:+.2f}] ms")
            print(f"      > warn       : {s['n_warn']}/{s['n']} ({s['pct_warn']:.1f}%)")
            print(f"      > error      : {s['n_error']}/{s['n']} ({s['pct_error']:.1f}%)")

        print(f"\n  {'─'*56}")
        print(f"  VERDICT : {icon} {r['verdict']}")
        if r['verdict'] == 'PASS':
            print("  Stabilité temporelle conforme au protocole.")
        elif r['verdict'] == 'WARN':
            print("  Quelques déviations isolées — données probablement utilisables.")
            print("  Consultez le CSV pour identifier les essais problématiques.")
        else:
            print("  ❌ PROBLÈMES DE TIMING SYSTÉMIQUES.")
            print(f"  >{ self.FAIL_PCT:.0f}% des essais dépassent le seuil d'erreur.")
            print("  → Vérifiez CPU, GPU, V-Sync et charge système.")
        print(f"{'='*60}\n")

    # =================================================================
    # SAUVEGARDE CSV
    # =================================================================
    def _save_csv(self, r):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename  = (
            f"SART_TimingQC_{r['participant']}_ses-{r['session']}"
            f"_{timestamp}.csv"
        )
        filepath = os.path.join(self.qc_dir, filename)

        rows = []

        # En-tête général
        rows.append(['SART Timing QC Report'])
        rows.append(['Participant', r['participant'],
                     'Session', r['session'],
                     'Trials', r['n_trials'],
                     'Verdict', r['verdict']])
        rows.append(['Frame rate (Hz)', r['frame_rate_hz'],
                     'Frame dur (ms)', r['frame_dur_ms'],
                     'Warn threshold (ms)', r['warn_ms'],
                     'Error threshold (ms)', r['error_ms']])
        rows.append([])

        # Stats par métrique
        header = ['Metric', 'N', 'Mean', 'SD', 'Median',
                  'Min', 'Max', 'AbsMax',
                  'N_warn', 'Pct_warn', 'N_error', 'Pct_error']
        rows.append(header)

        for key in ('digit', 'mask', 'total', 'soa'):
            s = r['stats'][key]
            if s.get('n', 0) == 0:
                continue
            rows.append([
                s['label'], s['n'],
                s['mean'], s['sd'], s['median'],
                s['min'], s['max'], s['abs_max'],
                s['n_warn'], s['pct_warn'],
                s['n_error'], s['pct_error'],
            ])

        rows.append([])

        # Détail essai par essai
        rows.append(['--- Trial Detail ---'])
        rows.append(['trial', 'phase', 'digit', 'condition',
                     'actual_digit_ms', 'actual_mask_ms', 'actual_total_ms',
                     'digit_error_ms', 'mask_error_ms', 'total_error_ms'])
        for t in self.timing_log:
            rows.append([
                t.get('trial', ''),
                t.get('phase', ''),
                t.get('digit', ''),
                t.get('condition', ''),
                t.get('actual_digit_ms', ''),
                t.get('actual_mask_ms', ''),
                t.get('actual_total_ms', ''),
                t.get('digit_error_ms', ''),
                t.get('mask_error_ms', ''),
                t.get('total_error_ms', ''),
            ])

        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerows(rows)
            self.logger.ok(f"[QC] Rapport CSV → {filepath}")
        except Exception as e:
            self.logger.warn(f"[QC] Erreur sauvegarde CSV : {e}")