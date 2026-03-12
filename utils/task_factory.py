# task_factory.py

from tasks.sart import sart


def create_task(config, win):
    base_kwargs = {
        'win':           win,
        'nom':           config['nom'],
        'enregistrer':   config['enregistrer'],
        'screenid':      config['screenid'],
        'parport_actif': config['parport_actif'],
        'mode':          config['mode'],
        'session':       config['session'],
    }

    task_config = config['tache']



    if task_config == 'sart':
        sart_config = {
            'mode':              config['run_type'],       # 'training' | 'classic'
            'participant_id':    config['nom'],
            'session':           config['session'],
            'target_digit':      config.get('target_digit', 3),
            'response_key':      config.get('response_key', 'space'),
            'n_trials_training': config.get('n_trials_training', 18),
            'trial_file':        config.get('trial_file', 'SART_trials_McGill.xlsx'),
            'isi_range':         config.get('isi_range', (0.300, 0.700)),
            'data_dir':          f"data/sart/{config['nom']}",
        }
        return sart(win=win, config=sart_config)

    else:
        print("Tâche inconnue.")
        return None