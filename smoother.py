import json
import argparse
import logging
from OneEuroFilter import OneEuroFilter
import matplotlib.pyplot as plt

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def smooth_animation_data(input_path, output_path, mincutoff=1.0, beta=0.0, preview=False):
    """
    Charge les données d'animation, applique un filtre One-Euro, et sauvegarde les données lissées.
    Optionnellement, affiche un graphique de comparaison.
    """
    setup_logging()
    logging.info(f"Chargement du fichier d'entrée: {input_path}")
    try:
        with open(input_path, 'r') as f:
            raw_data = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logging.error(f"Impossible de lire le fichier JSON d'entrée: {e}")
        return

    smoothed_data = {}
    video_fps = 30  # Hypothèse raisonnable pour la fréquence du filtre

    # Pour la visualisation
    preview_raw_vals = []
    preview_smoothed_vals = []

    for person_id, track_data in raw_data.items():
        logging.info(f"Traitement de la personne ID: {person_id}")
        
        filters = {}
        first_valid_frame = next((frame for frame in track_data if frame['landmarks']), None)
        if not first_valid_frame: continue

        for lmk_type, lmk_list in first_valid_frame['landmarks'].items():
            if lmk_list:
                # Correction du nom du paramètre: min_cutoff -> mincutoff
                filters[lmk_type] = {
                    'x': [OneEuroFilter(freq=video_fps, mincutoff=mincutoff, beta=beta) for _ in lmk_list],
                    'y': [OneEuroFilter(freq=video_fps, mincutoff=mincutoff, beta=beta) for _ in lmk_list],
                    'z': [OneEuroFilter(freq=video_fps, mincutoff=mincutoff, beta=beta) for _ in lmk_list]
                }

        new_track_data = []
        for frame_data in track_data:
            frame_num = frame_data['frame']
            timestamp = frame_num / video_fps
            new_landmarks = {}

            for lmk_type, lmk_list in frame_data['landmarks'].items():
                if lmk_list and lmk_type in filters:
                    smoothed_lmk_list = []
                    for i, lmk in enumerate(lmk_list):
                        smooth_x = filters[lmk_type]['x'][i](lmk['x'], timestamp)
                        smooth_y = filters[lmk_type]['y'][i](lmk['y'], timestamp)
                        smooth_z = filters[lmk_type]['z'][i](lmk['z'], timestamp)
                        smoothed_lmk_list.append({'x': smooth_x, 'y': smooth_y, 'z': smooth_z, 'visibility': lmk['visibility']})
                        
                        # Collecter des données pour le preview (ex: poignet droit, axe X)
                        if preview and person_id == next(iter(raw_data)) and lmk_type == 'pose' and i == 16:
                            preview_raw_vals.append(lmk['x'])
                            preview_smoothed_vals.append(smooth_x)

                    new_landmarks[lmk_type] = smoothed_lmk_list
            
            new_track_data.append({'frame': frame_num, 'landmarks': new_landmarks})
        
        smoothed_data[person_id] = new_track_data

    logging.info(f"Sauvegarde des données lissées dans: {output_path}")
    with open(output_path, 'w') as f: json.dump(smoothed_data, f, indent=4)
    logging.info("Lissage terminé avec succès.")

    # Afficher le graphique de prévisualisation si demandé
    if preview and preview_raw_vals:
        plt.figure(figsize=(15, 6))
        plt.plot(preview_raw_vals, 'r-', alpha=0.5, label='Données Brutes')
        plt.plot(preview_smoothed_vals, 'b-', label='Données Lissées')
        plt.title("Comparaison Lissage (Coordonnée X du poignet droit)")
        plt.xlabel("Image")
        plt.ylabel("Position (X)")
        plt.legend()
        plt.grid(True)
        logging.info("Affichage du graphique de comparaison. Fermez la fenêtre pour continuer.")
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Applique un lissage (One-Euro Filter) aux données d'animation JSON.")
    parser.add_argument("--input_json", required=True, help="Fichier JSON d'entrée.")
    parser.add_argument("--output_json", required=True, help="Fichier JSON de sortie.")
    parser.add_argument("--mincutoff", type=float, default=1.0, help="Fréquence de coupure minimale.")
    parser.add_argument("--beta", type=float, default=0.0, help="Paramètre beta du filtre.")
    parser.add_argument("--preview", action="store_true", help="Affiche un graphique comparant les données brutes et lissées.")

    args = parser.parse_args()
    smooth_animation_data(args.input_json, args.output_json, args.mincutoff, args.beta, args.preview)
