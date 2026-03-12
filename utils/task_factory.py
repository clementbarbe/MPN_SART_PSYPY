from tasks.temporaljudgement import TemporalJudgement

def create_task(config, win):
    base_kwargs = {
        'win': win,
        'nom': config['nom'],
        'enregistrer': config['enregistrer'],
        'screenid': config['screenid'],
        'parport_actif': config['parport_actif'],
        'mode': config['mode'],
        'session': config['session'], 

    }

    task_config = config['tache']
    
    if task_config == 'TemporalJudgement':
        return TemporalJudgement(
            **base_kwargs,
            n_trials_base=config['n_trials_base'],
            n_trials_block=config['n_trials_block'],
            n_trials_training=config['n_trials_training'],
            run_type=config['run_type']            
        )
    
    else:
        print("Tâche inconnue.")
        return None