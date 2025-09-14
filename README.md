# MotionExtract 3D

Ce projet est une pipeline complète pour extraire le mouvement d'un personnage depuis une vidéo, le traiter, et l'exporter en tant qu'animation 3D au format FBX. Il utilise MediaPipe pour la détection de pose et YoloV8 pour la détection de personne (bien que le modèle `.pt` soit ignoré dans ce dépôt).

## Pipeline de Traitement

Le processus se déroule en plusieurs étapes, orchestrées par `run_pipeline.py`:

1.  **Extraction (`extract_motion.py`)**: Analyse la vidéo d'entrée pour détecter les points de repère du corps humain (pose) et sauvegarde les données brutes dans un fichier JSON.
2.  **Lissage (`smoother.py`)**: Applique un filtre pour lisser les données de mouvement brutes et réduire les saccades.
3.  **Calcul des Rotations (`calculate_rotations.py`)**: Convertit les positions des points de repère en rotations d'articulations, une étape nécessaire pour l'animation de squelette.
4.  **Exportation FBX (`export_to_fbx.py`)**: Crée un fichier FBX contenant un squelette et l'animation correspondante, prêt à être importé dans un logiciel 3D.

## Installation

1.  Assurez-vous que Python 3.8+ est installé.
2.  Installez les dépendances requises :
    ```bash
    pip install -r requirements.txt
    ```
3.  (Optionnel) Téléchargez un modèle de détection d'objet comme `yolov8n.pt` si vous souhaitez l'utiliser dans le script d'extraction.

## Utilisation

Pour lancer la pipeline complète, utilisez le script `run_pipeline.py`.

```bash
python run_pipeline.py --video_path "chemin/vers/votre/video.mp4" --output_name "nom_sortie"
```

Le script générera plusieurs fichiers intermédiaires (`.json`) et le fichier final `nom_sortie_animation.fbx` dans le dossier `output/`.