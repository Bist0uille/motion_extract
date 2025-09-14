import json
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (12, 14),
    (14, 16), (16, 18), (16, 20), (16, 22), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (24, 26), (26, 28), (28, 30),
    (28, 32), (31, 32)
]

def visualize_animation(input_path):
    setup_logging()
    logging.info(f"Chargement des données depuis: {input_path}")
    with open(input_path, 'r') as f:
        data = json.load(f)

    person_id = next(iter(data))
    track_data = data[person_id]
    logging.info(f"Visualisation de l'animation pour la personne ID: {person_id}")

    all_x, all_y, all_z = [], [], []
    frames_data = []
    for frame_data in track_data:
        if frame_data['landmarks'] and frame_data['landmarks'].get('pose'):
            landmarks = frame_data['landmarks']['pose']
            frames_data.append(landmarks)
            for lm in landmarks:
                all_x.append(lm['x'])
                # Inversion Y et Z pour une vue Z-up
                all_y.append(-lm['z']) # Y devient -Z
                all_z.append(lm['y'])  # Z devient Y

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    z_min, z_max = min(all_z), max(all_z)
    
    # Rendre les axes cubiques
    max_range = np.array([x_max-x_min, y_max-y_min, z_max-z_min]).max() / 2.0
    mid_x = (x_max+x_min) * 0.5
    mid_y = (y_max+y_min) * 0.5
    mid_z = (z_max+z_min) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X (Droite)')
    ax.set_ylabel('Y (Avant)')
    ax.set_zlabel('Z (Haut)')
    ax.set_title(f"Animation 3D - Personne {person_id}")
    ax.view_init(elev=15., azim=-75) # Angle de vue

    def animate(frame_index):
        ax.cla()
        landmarks = frames_data[frame_index]
        # Inversion Y/Z et direction pour la visualisation
        pts = np.array([[lm['x'], -lm['z'], lm['y']] for lm in landmarks])

        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='blue')

        for i, j in POSE_CONNECTIONS:
            if i < len(pts) and j < len(pts):
                ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], [pts[i, 2], pts[j, 2]], 'r-')
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_xlabel('X (Droite)')
        ax.set_ylabel('Y (Avant)')
        ax.set_zlabel('Z (Haut)')
        ax.set_title(f"Animation 3D - Personne {person_id} (Image {frame_index})")

    anim = FuncAnimation(fig, animate, frames=len(frames_data), interval=33)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualise une animation de squelette 3D à partir d'un fichier JSON.")
    parser.add_argument("--input_json", required=True, help="Fichier JSON d'entrée (données lissées).")
    args = parser.parse_args()
    visualize_animation(args.input_json)
