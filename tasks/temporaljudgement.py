import random
import gc
import os
import glob
from psychopy import visual, core
from utils.base_task import BaseTask
from tasks.qc.qc_temporal import qc_temporaljudgement


class TemporalJudgement(BaseTask):
    """
    Tâche de jugement de délai temporel entre une action et un stimulus visuel.
    """

    def __init__(self, win, nom, session='01', mode='fmri', run_type='base',
                 n_trials_base=72, n_trials_block=24, n_trials_training=12,
                 delays_ms=(200, 300, 400, 500, 600, 700),
                 response_options=(100, 200, 300, 400, 500, 600, 700, 800),
                 stim_isi_range=(1500, 2500),
                 enregistrer=True, eyetracker_actif=False, parport_actif=True,
                 **kwargs):

        super().__init__(
            win=win,
            nom=nom,
            session=session,
            task_name="Temporal Judgement",
            folder_name="temporal_judgement",
            eyetracker_actif=eyetracker_actif,
            parport_actif=parport_actif,
            enregistrer=enregistrer,
            et_prefix='TJ'
        )

        self.mode = mode.lower()
        self.run_type = run_type.lower()

        self.n_trials_base = n_trials_base
        self.n_trials_block = n_trials_block
        self.n_trials_training = n_trials_training

        self.delays_ms = list(delays_ms)
        self.response_values_ms = list(response_options)
        self.stim_isi_range = (stim_isi_range[0] / 1000.0, stim_isi_range[1] / 1000.0)

        self.global_records = []
        self.current_trial_idx = 0
        self.current_phase = 'setup'

        self._detect_display_scaling()
        self._measure_frame_rate()
        self._define_ttl_codes()
        self._setup_key_mapping()
        self._setup_task_stimuli()
        self._init_incremental_file(suffix=f"_{self.run_type}")

        self.logger.ok(
            f"TemporalJudgement init | Mode: {self.run_type} | "
            f"Frame Rate: {self.frame_rate:.2f} Hz"
        )

    # =========================================================================
    # INITIALISATION
    # =========================================================================

    def _detect_display_scaling(self):
        if self.win.size[1] > 1200:
            self.pixel_scale = 2.0
            self.logger.log(f"High-res display ({self.win.size}). Scale: x2.0")
        else:
            self.pixel_scale = 1.0
            self.logger.log(f"Standard display ({self.win.size}). Scale: x1.0")
        self.x_spacing_scale = 1.14

    def _measure_frame_rate(self):
        self.logger.log("Measuring frame rate...")
        self.frame_rate = self.win.getActualFrameRate(
            nIdentical=10, nMaxFrames=100, threshold=1
        )
        if self.frame_rate is None:
            self.frame_rate = 60.0
            self.logger.warn("Frame rate not detected, defaulting to 60.0 Hz")
        else:
            self.logger.ok(f"Frame rate: {self.frame_rate:.2f} Hz")

        self.frame_duration_s = 1.0 / self.frame_rate
        self.frame_tolerance_s = 0.75 / self.frame_rate
        self.logger.log(f"Frame tolerance: {self.frame_tolerance_s * 1000:.2f} ms")

    def _define_ttl_codes(self):
        self.codes = {
            'start_exp': 255,
            'rest_start': 200,
            'rest_end': 201,
            'trial_active': 110,
            'trial_passive': 111,
            'action_bulb': 120,
            'bulb_on': 130,
            'response_prompt': 135,
            'response_given': 140,
            'timeout': 199,
            'crisis_prompt': 150,
            'crisis_start': 151,
            'crisis_end': 152,
            'crisis_valid_prompt': 153,
            'crisis_res_success': 154,
            'crisis_res_fail': 155,
            'crisis_retry_yes': 156,
            'crisis_retry_no': 157
        }

    def _setup_key_mapping(self):
        if self.mode == 'fmri':
            self.key_action = 'b'
            self.keys_responses = ['d', 'n', 'z', 'e', 'b', 'y', 'g', 'r']
            self.key_trigger = 't'
        else:
            self.key_action = 'y'
            self.keys_responses = ['a', 'z', 'e', 'r', 'y', 'u', 'i', 'o']
            self.key_trigger = 't'

        self.response_key_to_ms = {
            key: ms for key, ms in zip(self.keys_responses, self.response_values_ms)
        }

    def _setup_task_stimuli(self):
        bulb_size = (0.45 * 0.9, 0.9 * 0.9)
        bulb_pos = (0.0, 0.0)

        img_off = os.path.join(self.img_dir, 'bulbof.png')
        img_on = os.path.join(self.img_dir, 'bulbon.png')

        if os.path.exists(img_off) and os.path.exists(img_on):
            self.bulb_off_img = visual.ImageStim(
                self.win, image=img_off, size=bulb_size, pos=bulb_pos
            )
            self.bulb_on_img = visual.ImageStim(
                self.win, image=img_on, size=bulb_size, pos=bulb_pos
            )
        else:
            self.logger.warn("Bulb images not found, using circles.")
            self.bulb_off_img = visual.Circle(self.win, radius=0.2, fillColor='grey')
            self.bulb_on_img = visual.Circle(self.win, radius=0.2, fillColor='yellow')

        self.colored_bar = visual.Rect(
            win=self.win, width=0.15, height=0.04, pos=(0.0, -0.5)
        )

        self.response_title = visual.TextStim(
            self.win, text="Combien de ms avez-vous perçu ?",
            color='white', height=0.05, pos=(0, 0.3)
        )
        self.response_options_text = visual.TextStim(
            self.win,
            text="1: 100 | 2: 200 | 3: 300 | 4: 400 | 5: 500 | 6: 600 | 7: 700 | 8: 800",
            color='white', height=0.05, pos=(0, 0.05)
        )
        self.response_instr = visual.TextStim(
            self.win, text="Répondez avec les 8 boutons",
            color='white', height=0.045, pos=(0, -0.2)
        )

        base_positions = [-0.35, -0.255, -0.15, -0.05, 0.055, 0.16, 0.26, 0.36]
        self.underline_x_positions = [x * self.x_spacing_scale for x in base_positions]
        self.underline_y_line = -0.055

        self.logger.log("Stimuli loaded.")

    # =========================================================================
    # LOGGING
    # =========================================================================

    def log_trial_event(self, event_type, **kwargs):
        current_time = self.task_clock.getTime()

        if self.eyetracker_actif:
            self.EyeTracker.send_message(
                f"PHASE_{self.current_phase.upper()}_"
                f"TRIAL_{self.current_trial_idx:03d}_{event_type.upper()}"
            )

        entry = {
            'participant': self.nom,
            'session': self.session,
            'phase': self.current_phase,
            'trial': self.current_trial_idx,
            'time_s': round(current_time, 5),
            'event_type': event_type
        }
        entry.update(kwargs)
        self.global_records.append(entry)

    # =========================================================================
    # CORE TASK LOGIC
    # =========================================================================

    def draw_lightbulb(self, base_color, bulb_on=False):
        self.colored_bar.fillColor = base_color
        self.colored_bar.lineColor = base_color
        self.colored_bar.draw()
        bulb = self.bulb_on_img if bulb_on else self.bulb_off_img
        bulb.draw()

    def run_trial(self, trial_index, total_trials, condition, delay_ms, feedback=False):
        """
        Exécute un essai complet avec timing sub-millisecondes.

        """
        self.should_quit()

        # =================================================================
        # CRITICAL TIMING: DISABLE GC
        # =================================================================
        gc.disable()

        self.current_trial_idx = trial_index
        base_color = '#00FF00' if condition == 'active' else '#FF0000'

        # --- Phase 1: Trial Start ---
        self.log_trial_event(
            'trial_start', condition=condition,
            delay_target_ms=delay_ms, feedback_mode=feedback
        )
        trigger_code = (
            self.codes['trial_active'] if condition == 'active'
            else self.codes['trial_passive']
        )

        self.fixation.draw()
        self.win.callOnFlip(self.ParPort.send_trigger, trigger_code)
        self.win.flip()
        core.wait(0.5)

        # --- Phase 2: Action or Auto ---
        self.draw_lightbulb(base_color=base_color, bulb_on=False)
        self.win.flip()

        action_time = self.task_clock.getTime()

        self.flush_keyboard()
        while True:
            keys = self.get_keys(key_list=[self.key_action])
            if keys:
                # Même horloge que la boucle de délai = pas de décalage
                action_time = self.task_clock.getTime()
                self.ParPort.send_trigger(self.codes['action_bulb'])
                self.log_trial_event('action_performed', action_key=keys[0].name)
                break
            pass

        # --- Phase 3: Precise Delay (Critical Timing) ---
        target_light_time = action_time + (delay_ms / 1000.0)

        while self.task_clock.getTime() < (target_light_time - self.frame_tolerance_s):
            pass

        # --- Phase 4: Bulb ON (synchronized flip) ---
        self.draw_lightbulb(base_color=base_color, bulb_on=True)
        self.win.callOnFlip(self.ParPort.send_trigger, self.codes['bulb_on'])

        flip_timestamps = {}

        def _capture_flip_time():
            flip_timestamps['bulb_on'] = self.task_clock.getTime()

        self.win.callOnFlip(_capture_flip_time)
        self.win.flip()

        bulb_on_time = flip_timestamps.get('bulb_on', self.task_clock.getTime())
        actual_delay = (bulb_on_time - action_time) * 1000
        error_ms = actual_delay - delay_ms

        if abs(error_ms) > (self.frame_duration_s * 1000 * 1.5):
            self.logger.warn(
                f"TIMING WARNING Trial {trial_index}: "
                f"error={error_ms:.2f}ms (>{self.frame_duration_s*1000*1.5:.1f}ms 1.5 frame)"
            )

        self.log_trial_event('bulb_lit', actual_delay_ms=actual_delay, error_ms=error_ms)

        wait_duration = random.uniform(1.2, 1.8)
        core.wait(wait_duration)
        self.win.flip()

        # --- Phase 5: Response Prompt ---
        t0_response = self.task_clock.getTime()
        self.log_trial_event('response_prompt_shown')
        self.ParPort.send_trigger(self.codes['response_prompt'])

        self.response_title.draw()
        self.response_options_text.draw()
        self.response_instr.draw()
        self.win.flip()

        resp_keys = self.wait_keys(
            key_list=self.keys_responses,
            max_wait=5.0
        )

        rt = None
        response_ms = None

        # --- Trial summary record for incremental save ---
        trial_summary = {
            'participant': self.nom,
            'session': self.session,
            'phase': self.current_phase,
            'trial': trial_index,
            'condition': condition,
            'delay_target_ms': delay_ms,
            'actual_delay_ms': round(actual_delay, 3),
            'timing_error_ms': round(error_ms, 3),
            'feedback': feedback,
        }

        if resp_keys:
            resp_key_name = resp_keys[0].name
            timestamp_key = resp_keys[0].tDown

            self.ParPort.send_trigger(self.codes['response_given'])

            rt = timestamp_key - t0_response
            response_ms = self.response_key_to_ms.get(resp_key_name)
            idx_user = self.keys_responses.index(resp_key_name)

            # --- Feedback ---
            if feedback:
                is_correct = (response_ms == delay_ms)
                user_bar_color = 'green' if is_correct else 'yellow'
                msg_text = (
                    "Bonne réponse !" if is_correct
                    else f"Réponse correcte : {delay_ms} ms"
                )
                msg_color = 'green' if is_correct else 'red'

                self.response_title.draw()
                self.response_options_text.draw()

                current_line_width = 5 * self.pixel_scale
                user_line = visual.Line(
                    self.win,
                    start=(self.underline_x_positions[idx_user] - 0.04,
                           self.underline_y_line),
                    end=(self.underline_x_positions[idx_user] + 0.04,
                         self.underline_y_line),
                    lineColor=user_bar_color,
                    lineWidth=current_line_width
                )
                user_line.draw()

                if not is_correct:
                    try:
                        idx_correct = self.response_values_ms.index(delay_ms)
                        thick_line_width = 6 * self.pixel_scale
                        correct_line = visual.Line(
                            self.win,
                            start=(self.underline_x_positions[idx_correct] - 0.04,
                                   self.underline_y_line),
                            end=(self.underline_x_positions[idx_correct] + 0.04,
                                 self.underline_y_line),
                            lineColor='red',
                            lineWidth=thick_line_width
                        )
                        correct_line.draw()
                    except ValueError:
                        pass

                fb_text = visual.TextStim(
                    self.win, text=msg_text, color=msg_color,
                    height=0.05, pos=(0, -0.2)
                )
                fb_text.draw()
                self.win.flip()
                core.wait(1.0)

            else:
                underline = visual.Line(
                    self.win,
                    start=(self.underline_x_positions[idx_user] - 0.04,
                           self.underline_y_line),
                    end=(self.underline_x_positions[idx_user] + 0.04,
                         self.underline_y_line),
                    lineColor='yellow',
                    lineWidth=5 * self.pixel_scale
                )
                self.response_title.draw()
                self.response_options_text.draw()
                self.response_instr.draw()
                underline.draw()
                self.win.flip()
                core.wait(0.6)

            self.log_trial_event(
                'response_given',
                response_key=resp_key_name,
                response_ms=response_ms,
                rt_s=rt
            )

            fb_str = f"| FB: {'Yes' if feedback else 'No':<3}"
            self.logger.log(
                f"Trial {trial_index:>2}/{total_trials:<2} | "
                f"{condition.upper():<7} | "
                f"Target: {delay_ms:>3}ms | "
                f"Answer: {str(response_ms):>4}ms | "
                f"RT: {rt:.3f}s {fb_str}"
            )

            # Complete trial summary
            trial_summary.update({
                'response_key': resp_key_name,
                'response_ms': response_ms,
                'rt_s': round(rt, 5),
                'timeout': False
            })

        else:
            # --- TIMEOUT ---
            self.ParPort.send_trigger(self.codes['timeout'])
            self.log_trial_event('response_timeout')

            too_slow = visual.TextStim(
                self.win, text="Temps de réponse écoulé",
                color='red', height=0.1
            )
            too_slow.draw()
            self.win.flip()
            core.wait(0.8)

            self.logger.warn(
                f"Trial {trial_index:>2}/{total_trials:<2} | "
                f"{condition.upper():<7} | "
                f"Target: {delay_ms:>3}ms | TIMEOUT"
            )

            trial_summary.update({
                'response_key': None,
                'response_ms': None,
                'rt_s': None,
                'timeout': True
            })

        # =================================================================
        # CRITICAL TIMING END: RE-ENABLE GC
        # =================================================================
        gc.enable()
        gc.collect()

        self.save_trial_incremental(trial_summary)

        # --- ITI ---
        isi = random.uniform(*self.stim_isi_range)
        self.fixation.draw()
        self.win.flip()
        core.wait(isi)

        self.log_trial_event('trial_end', isi_duration=isi)

        return True

    # =========================================================================
    # TRIAL GENERATION
    # =========================================================================

    def build_trials(self, n_trials, training=False):
        conditions = ['active'] if training else ['active', 'passive']

        if training:
            unique_types = [(c, d) for c in conditions for d in self.delays_ms]
            n_full_repeats = n_trials // len(unique_types)
            remainder = n_trials % len(unique_types)
            trials = unique_types * n_full_repeats
            if remainder > 0:
                trials.extend(random.sample(unique_types, remainder))
            random.shuffle(trials)
            return trials

        def build_pool():
            n_per_condition = n_trials // 2
            remainder = n_trials % 2
            pool = []
            for cond in conditions:
                n_for_cond = n_per_condition + (
                    1 if cond == 'active' and remainder else 0
                )
                n_per_delay = n_for_cond // len(self.delays_ms)
                delay_remainder = n_for_cond % len(self.delays_ms)
                for delay in self.delays_ms:
                    count = n_per_delay + (1 if delay_remainder > 0 else 0)
                    if delay_remainder > 0:
                        delay_remainder -= 1
                    pool.extend([(cond, delay)] * count)
            return pool

        def count_recent(trial_list, condition=None, delay=None):
            """Count consecutive matching trials from the end of the list."""
            if not trial_list:
                return 0
            count = 0
            for trial in reversed(trial_list):
                match = False
                if condition is not None and trial[0] == condition:
                    match = True
                if delay is not None and trial[1] == delay:
                    match = True
                if match:
                    count += 1
                else:
                    break
            return count

        def is_valid(trial_list, candidate):
            cond, delay = candidate
            if count_recent(trial_list, condition=cond) >= 2:
                return False
            if count_recent(trial_list, delay=delay) >= 1:
                return False
            return True

        def get_best_candidates(pool, trial_list):
            if not trial_list:
                return pool[:]
            valid = [t for t in pool if is_valid(trial_list, t)]
            if not valid:
                return []

            def score_candidate(candidate):
                cond, delay = candidate
                cond_penalty = count_recent(trial_list, condition=cond) * 10
                delay_penalty = count_recent(trial_list, delay=delay) * 20
                diversity_bonus = 0
                if trial_list[-1][0] != cond:
                    diversity_bonus += 5
                if trial_list[-1][1] != delay:
                    diversity_bonus += 3
                return -(cond_penalty + delay_penalty - diversity_bonus)

            valid.sort(key=score_candidate, reverse=True)
            return valid

        def build_sequence(pool, max_attempts=50):
            for attempt in range(max_attempts):
                sequence = []
                remaining = pool[:]
                random.shuffle(remaining)
                while len(sequence) < n_trials:
                    candidates = get_best_candidates(remaining, sequence)
                    if not candidates:
                        if len(sequence) < 3:
                            break
                        for _ in range(min(3, len(sequence))):
                            removed = sequence.pop()
                            remaining.append(removed)
                        random.shuffle(remaining)
                        continue
                    top_n = min(5, len(candidates))
                    chosen = random.choice(candidates[:top_n])
                    sequence.append(chosen)
                    remaining.remove(chosen)
                if len(sequence) == n_trials:
                    return sequence
            return None

        max_restarts = 20
        for attempt in range(max_restarts):
            pool = build_pool()
            result = build_sequence(pool)
            if result:
                return result

        self.logger.warn(
            f"Constraints hard to satisfy for {n_trials} trials. Using simple shuffle."
        )
        pool = build_pool()
        random.shuffle(pool)
        return pool

    def run_trial_block(self, n_trials, block_name, phase_tag, feedback):
        self.current_phase = phase_tag
        self.log_trial_event('block_start', block_name=block_name, feedback_mode=feedback)
        self.logger.log(f"--- Block Start: {block_name} ({n_trials} trials) ---")

        trials = self.build_trials(n_trials, training=(phase_tag == 'training'))
        total_trials = len(trials)

        for i, (cond, delay) in enumerate(trials, start=1):
            self.run_trial(i, total_trials, cond, delay, feedback=feedback)

        self.log_trial_event('block_end', block_name=block_name)
        self.logger.log(f"--- Block End: {block_name} ---")

    # =========================================================================
    # CRISIS VALIDATION WINDOW
    # =========================================================================

    def show_crisis_validation_window(self):
        self.current_phase = 'crisis_validation'
        self.current_trial_idx = None
        self.logger.log("=== Entering Crisis Validation Window ===")

        loop_crisis = True

        while loop_crisis:
            # --- 1. Launch Prompt ---
            msg_launch = visual.TextStim(
                self.win, text='Appuyez pour démarrer la crise', height=0.08
            )
            msg_launch.draw()
            self.win.flip()

            self.log_trial_event('crisis_prompt_start')
            self.ParPort.send_trigger(self.codes['crisis_prompt'])

            keys = self.wait_keys(key_list=self.keys_responses)

            # --- 2. Crisis Action ---
            self.log_trial_event('crisis_action_started', trigger_key=keys[0].name)
            self.ParPort.send_trigger(self.codes['crisis_start'])

            self.fixation.draw()
            self.win.flip()
            core.wait(0.5)

            keys = self.wait_keys(key_list=self.keys_responses)

            self.ParPort.send_trigger(self.codes['crisis_end'])
            self.log_trial_event('crisis_action_ended', end_key=keys[0].name)

            # --- 3. Validation ---
            choice_text = visual.TextStim(
                self.win,
                text='[1-4] Crise réussie     [5-8] Crise échouée',
                height=0.08
            )
            choice_text.draw()
            self.win.flip()

            self.log_trial_event('crisis_validation_prompt')
            self.ParPort.send_trigger(self.codes['crisis_valid_prompt'])

            core.wait(0.2)
            keys = self.wait_keys(key_list=self.keys_responses)

            key_name = keys[0].name
            idx = self.keys_responses.index(key_name)
            success = idx < 4
            result_label = 'SUCCESS' if success else 'FAILED'

            trigger_code = (
                self.codes['crisis_res_success'] if success
                else self.codes['crisis_res_fail']
            )
            self.ParPort.send_trigger(trigger_code)

            self.log_trial_event(
                'crisis_result_chosen', result=result_label, key=key_name
            )
            self.logger.log(f"Crisis Outcome: {result_label}")

            confirmation = visual.TextStim(
                self.win, text=f"Résultat : {result_label}", height=0.08
            )
            confirmation.draw()
            self.win.flip()
            core.wait(1.0)

            # --- 4. Retry on Failure ---
            if not success:
                retry_text = visual.TextStim(
                    self.win,
                    text='Recommencer ?\n[1-4] Oui   [5-8] Non (Quitter)',
                    height=0.06
                )
                retry_text.draw()
                self.win.flip()

                keys = self.wait_keys(key_list=self.keys_responses)
                idx_retry = self.keys_responses.index(keys[0].name)

                if idx_retry >= 4:
                    self.log_trial_event(
                        'crisis_retry_decision', choice='no_retry_quit'
                    )
                    self.ParPort.send_trigger(self.codes['crisis_retry_no'])
                    self.should_quit(force_quit=True)
                else:
                    self.log_trial_event(
                        'crisis_retry_decision', choice='retry'
                    )
                    self.ParPort.send_trigger(self.codes['crisis_retry_yes'])
                    self.logger.log("Crisis Retry Selected.")
            else:
                loop_crisis = False

        self.log_trial_event('crisis_phase_end')

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    def run(self):
        finished_naturally = False
        saved_path = None

        try:
            # Instructions
            if self.run_type == 'training':
                instructions = (
                    "ENTRAINEMENT - Tâche de Jugement Temporel\n\n"
                    "Vous allez voir une ampoule.\n"
                    "Condition ACTIVE (barre verte) : "
                    "Appuyez sur le bouton pour l'allumer.\n"
                    "Condition PASSIVE (barre rouge) : "
                    "Elle s'allumera automatiquement.\n\n"
                    "Après un délai, estimez le temps perçu (100 à 800ms).\n\n"
                    f"Nombre d'essais : {self.n_trials_training} (avec feedback)\n\n"
                    "Appuyez sur 't' (trigger) pour commencer..."
                )
            else:
                instructions = (
                    "Tâche de Jugement Temporel\n\n"
                    "Condition ACTIVE (barre verte) : "
                    "Appuyez pour allumer l'ampoule.\n"
                    "Condition PASSIVE (barre rouge) : Attente passive.\n\n"
                    "Ensuite, évaluez le délai perçu (100 à 800ms).\n\n"
                    "Appuyez sur ESPACE pour continuer..."
                )

            self.show_instructions(instructions)
            self.wait_for_trigger()

            if self.run_type == 'training':
                self.logger.log(
                    f"Starting: TRAINING ({self.n_trials_training} trials)"
                )
                self.show_resting_state(duration_s=10.0)
                self.run_trial_block(
                    self.n_trials_training,
                    block_name="TRAINING",
                    phase_tag='training',
                    feedback=True
                )

            elif self.run_type == 'base':
                self.logger.log("Starting: FULL PROTOCOL")
                self.show_resting_state(duration_s=10.0)
                self.run_trial_block(
                    self.n_trials_base,
                    block_name="BASELINE",
                    phase_tag='base',
                    feedback=False
                )
                self.show_crisis_validation_window()
                self.run_trial_block(
                    self.n_trials_block,
                    block_name="POST_CRISIS",
                    phase_tag='run_standard',
                    feedback=False
                )

            else:
                self.logger.log("Starting: SHORT BLOCK")
                self.show_resting_state(duration_s=150.0)
                self.show_crisis_validation_window()
                self.run_trial_block(
                    self.n_trials_block,
                    block_name="STANDARD_BLOCK",
                    phase_tag='run_standard',
                    feedback=False
                )

            finished_naturally = True
            self.logger.ok("Task completed successfully.")

        except (KeyboardInterrupt, SystemExit):
            self.logger.warn("Manual interruption.")

        except Exception as e:
            self.logger.err(f"CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise

        finally:
            self.logger.log("Final save...")

            if self.eyetracker_actif:
                self.EyeTracker.stop_recording()
                self.EyeTracker.send_message("END_EXP")
                self.EyeTracker.close_and_transfer_data(self.data_dir)

            saved_path = self.save_data(
                data_list=self.global_records,
                filename_suffix=f"_{self.run_type}"
            )

            if saved_path and os.path.exists(saved_path):
                try:
                    qc_temporaljudgement(saved_path)
                except Exception as e:
                    self.logger.warn(f"QC generation failed: {e}")

            if finished_naturally:
                end_msg = "Fin de la session.\nMerci pour votre participation."
                self.show_instructions(end_msg)
                core.wait(3.0)