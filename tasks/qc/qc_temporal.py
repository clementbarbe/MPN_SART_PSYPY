import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def qc_temporaljudgement(csv_path):
    """
    Dashboard QC Final - Version "Stimuli & Réponses"
    1. Précision Technique
    2. Distribution ISI
    3. Jitter (Stimulus -> Échelle)
    4. Distribution des RÉPONSES (ms)
    5. Corrélation Target vs Réponse (R²)
    6. Distribution des STIMULI (Délais présentés)
    """
    if not os.path.exists(csv_path):
        print(f"QC Error: Fichier non trouvé {csv_path}")
        return

    print(f"--- Lancement du QC sur {os.path.basename(csv_path)} ---")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"QC Error: Impossible de lire le CSV. {e}")
        return

    # Dossier QC
    csv_dir = os.path.dirname(csv_path)
    qc_dir = os.path.join(csv_dir, 'qc')
    os.makedirs(qc_dir, exist_ok=True)

    # --- 1. RECONSTRUCTION ---
    df_trials = df[df['event_type'] == 'trial_start'][['phase', 'trial', 'condition', 'delay_target_ms']].copy()
    df_bulb = df[df['event_type'] == 'bulb_lit'][['phase', 'trial', 'error_ms', 'time_s']].rename(columns={'time_s': 't_bulb'})
    df_prompt = df[df['event_type'] == 'response_prompt_shown'][['phase', 'trial', 'time_s']].rename(columns={'time_s': 't_prompt'})
    df_resp = df[df['event_type'] == 'response_given'][['phase', 'trial', 'response_ms']].copy()
    df_isi = df[df['event_type'] == 'trial_end'][['isi_duration']].copy()

    df_full = pd.merge(df_trials, df_bulb, on=['phase', 'trial'], how='left')
    df_full = pd.merge(df_full, df_prompt, on=['phase', 'trial'], how='left')
    df_full = pd.merge(df_full, df_resp, on=['phase', 'trial'], how='left')

    df_full['jitter_ms'] = (df_full['t_prompt'] - df_full['t_bulb']) * 1000

    # --- 2. GRAPHIQUES ---
    plt.style.use('ggplot')
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"QC Report: {os.path.basename(csv_path)}", fontsize=16, fontweight='bold')
    
    colors = {'active': '#2ca02c', 'passive': '#d62728'} 

    # FIG 1: PRÉCISION GLOBALE
    ax = axes[0, 0]
    sns.histplot(df_full['error_ms'].dropna(), kde=True, ax=ax, color='steelblue')
    ax.set_title(f"1. Précision Technique\nMean: {df_full['error_ms'].mean():.2f}ms")
    ax.axvline(0, color='black', ls='--')

    # FIG 2: DISTRIBUTION ISI
    ax = axes[0, 1]
    if not df_isi.empty:
        sns.histplot(df_isi['isi_duration'].dropna(), kde=True, ax=ax, color='purple')
        ax.set_title(f"2. Distribution ISI\nRange: [{df_isi['isi_duration'].min():.1f}s - {df_isi['isi_duration'].max():.1f}s]")

    # FIG 3: JITTER (Stim -> Prompt)
    ax = axes[0, 2]
    jitter_vals = df_full['jitter_ms'].dropna()
    if not jitter_vals.empty:
        sns.histplot(jitter_vals, kde=True, ax=ax, color='chocolate')
        ax.set_title(f"3. Jitter (Stimulus -> Échelle)\nMoyenne: {jitter_vals.mean():.1f}ms")

    # FIG 4: DISTRIBUTION DES RÉPONSES (ms)
    ax = axes[1, 0]
    df_r = df_full.dropna(subset=['response_ms'])
    sns.countplot(x='response_ms', data=df_r, hue='condition', palette=colors, ax=ax)
    ax.set_title("4. Distribution des Réponses (ms)")
    ax.legend(title='Condition')

    # FIG 5: CORRÉLATION (R²)
    ax = axes[1, 1]
    df_corr = df_full.dropna(subset=['response_ms', 'delay_target_ms'])
    for cond, color in colors.items():
        sub = df_corr[df_corr['condition'] == cond]
        if len(sub) > 1:
            r = sub['delay_target_ms'].corr(sub['response_ms'])
            r2 = r**2 if not np.isnan(r) else 0
            sns.regplot(x='delay_target_ms', y='response_ms', data=sub, ax=ax, 
                        scatter_kws={'alpha':0.5}, color=color, label=f"{cond} ($R^2$={r2:.2f})")
    ax.plot([100, 800], [100, 800], 'k--', alpha=0.3)
    ax.set_title("5. Corrélation Target vs Réponse")
    ax.legend()

    # FIG 6: DISTRIBUTION DES STIMULI (DÉLAIS PRÉSENTÉS)
    ax = axes[1, 2]
    sns.countplot(x='delay_target_ms', data=df_full, hue='condition', palette=colors, ax=ax)
    total_resp = len(df_full[df_full['response_ms'].notna()])
    total_trials = len(df_full)
    ax.set_title(f"6. Stimuli présentés ({total_resp}/{total_trials} répondus)")
    ax.set_xlabel("Délai Cible (ms)")
    ax.set_ylabel("Nombre d'essais")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    png_name = os.path.basename(csv_path).replace('.csv', '_QC.png')
    save_path = os.path.join(qc_dir, png_name)
    plt.savefig(save_path, dpi=100)
    plt.close()
    
    print(f"QC Réussi : {save_path}")