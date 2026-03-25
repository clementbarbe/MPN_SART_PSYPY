# Hand Representation Task

Tâche de représentation corporelle basée sur le protocole Longo & Haggard (2010).  
Conçue pour l'évaluation neuropsychologique comportementale — CENIR, Institut du Cerveau (ICM).

## Principe

1. Une image indiquant un doigt et une zone précise apparaît à l'écran
2. Le participant pointe la zone indiquée **avant la fin de la barre de progression**
3. Le participant maintient sa position **sans bouger** jusqu'à l'image suivante
4. Une **photo** est prise automatiquement à la fin de chaque essai
5. Les images représentent naturellement une **main gauche** — un effet miroir est
   appliqué automatiquement pour la main droite

## Structure

| Niveau | Quantité | Description |
|--------|----------|-------------|
| Block | 1 | 100 essais au total |
| Miniblock | 10 par block | 10 essais chacun |
| Essai | 10 par miniblock | Les 10 positions, ordre aléatoire |

Les 10 positions couvrent **5 doigts × 2 zones** (proximale / distale).

## Paramètres

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `hand` | `droite` | Main testée — `droite` ou `gauche` |
| `n_blocks` | `1` | Nombre de blocks |
| `trial_duration` | `7.0` | Durée de la barre de progression (secondes) |
| `camera_index` | `0` | Index OpenCV de la webcam |
| `session` | `01` | Numéro de session |

## Effet miroir

Les images source représentent une main gauche.

| `hand` | Affichage |
|--------|-----------|
| `gauche` | Image telle quelle |
| `droite` | Image retournée horizontalement |

## Lancement

Prérequis

    Python 3.10+
    PsychoPy 2025.1.1
    opencv-python (cv2)

Images requises dans `images/` : `a1.png` … `a10.png`  
(5 doigts × 2 zones, main gauche, fond neutre)

## Données

Les résultats sont sauvegardés dans `data/hand_representation/` :

    *_final.csv              — fichier récapitulatif complet (fin de session)
    *_incremental.csv        — backup essai par essai (protection anti-crash)
    photos/                  — photos webcam au format JPEG

### Métriques enregistrées

- Identifiants : `participant`, `session`, `hand`, `flip_horiz`
- Position : `finger`, `zone`, `position_label`, `image_file`
- Temporel : `image_onset`, `capture_time_task`, `trial_duration`, `wall_timestamp`
- Arborescence : `block_number`, `miniblock_number`, `trial_in_miniblock`, `trial_in_block`
- Photo : `photo_filename`, `photo_path`

### Nomenclature des photos

    {participant}_{hand}_B{block}_M{miniblock}_T{trial}_{finger}_z{zone}_{timestamp}.jpg

    ex : SUJ01_droite_B01_M03_T027_index_z2_20250612_143201_000123.jpg

## Référence

> Longo, M. R., & Haggard, P. (2010).  
> **An implicit body representation underlying human position sense.**  
> *Proceedings of the National Academy of Sciences*, 107(26), 11727–11732.  
> https://doi.org/10.1073/pnas.1003483107

## Auteur

Clément BARBE — CENIR, Institut du Cerveau (ICM), Paris
