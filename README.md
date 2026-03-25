# Sustained Attention to Response Task (SART)

Tâche d'attention soutenue basée sur le protocole Robertson et al. (1997) — version McGill.  
Conçue pour l'évaluation neuropsychologique comportementale — CENIR, Institut du Cerveau (ICM).

## Principe

1. Une série de chiffres (1–9) apparaît au centre de l'écran
2. Chaque chiffre est présenté **250 ms** puis immédiatement masqué **900 ms** (SOA = 1150 ms)
3. **GO** : le participant appuie sur ESPACE pour tout chiffre sauf 3
4. **NO-GO** : le participant inhibe sa réponse lorsque le chiffre **3** apparaît
5. Les chiffres varient en taille (4 niveaux) pour maintenir l'engagement attentionnel

## Modes

| Mode | Essais | Feedback | Description |
|------|--------|----------|-------------|
| `full` | 20 + 225 | Entraînement seulement | Entraînement → écran de transition → tâche complète |
| `training` | 20 | Oui | Entraînement seul (18 GO + 2 NO-GO), ordre fixe |
| `test` | 225 | Non | Tâche complète seule (200 GO + 25 NO-GO), ordre fixe |

## Paramètres

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `mode` | `full` | Mode de passation |
| `target_digit` | `3` | Chiffre cible (NO-GO) |
| `response_key` | `space` | Touche de réponse |
| `training_feedback` | `True` | Feedback visuel pendant l'entraînement |
| `session` | `01` | Numéro de session |

## Lancement

Prérequis

    Python 3.10+
    PsychoPy 2025.1.1
    openpyxl
    numpy / pandas

Fichier de séquence test requis : `SART_trials_McGill.xlsx`  
(colonnes : `trial`, `digit`, `isnogo`, `size`)

## Données

Les résultats sont sauvegardés dans `data/sart/` :

    McGill_SART_Raw_Data_*_{mode}_{timestamp}.xlsx — fichier final (feuilles Training / Test / All_Trials)
    *_incremental.csv — backup trial par trial (protection anti-crash)
    qc/SART_TimingQC_*_{timestamp}.csv — rapport de qualité temporelle

### Métriques enregistrées

- Type de réponse : `Go Success`, `Go Ambiguous`, `Go Anticipatory`, `NoGo Success`, `NoGo Failure`, `Omission`
- RT et latence (ms), type de latence (0–3)
- Compteurs cumulatifs : `countGo`, `countNoGo`, `countValidGo`, `countAnticipatory`, `correctSuppressions`, `incorrectSuppressions`

## Contrôle qualité timing

Un rapport QC est généré automatiquement après chaque session (≥ 25 essais requis).  
Le verdict porte sur la stabilité du digit (250 ms) et du masque (900 ms) uniquement :

    PASS  : <2%  des essais dépassent le seuil d'erreur (±2.5 frames)
    WARN  : 2–10% des essais dépassent le seuil d'erreur
    FAIL  : >10% des essais dépassent le seuil d'erreur
    N/A   : échantillon insuffisant (< 25 essais)

## Auteur

Clément BARBE — CENIR, Institut du Cerveau (ICM), Paris
