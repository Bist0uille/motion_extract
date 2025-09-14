import json
import argparse
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Définition du squelette et de la pose T de référence
# Les index correspondent aux landmarks de MediaPipe Pose
SKELETON_BONES = {
    'hips': (24, 23),
    'spine': (12, 11),
    'neck': (12, 0), # Approximatif
    'head': (0, 0),

    'left_upper_arm': (11, 13),
    'left_lower_arm': (13, 15),
    'right_upper_arm': (12, 14),
    'right_lower_arm': (14, 16),

    'left_upper_leg': (23, 25),
    'left_lower_leg': (25, 27),
    'right_upper_leg': (24, 26),
    'right_lower_leg': (26, 28),
}

def get_vector(landmarks, start_idx, end_idx):
    """Calcule un vecteur normalisé à partir de deux points landmarks."""
    start = np.array([landmarks[start_idx][c] for c in 'xyz'])
    end = np.array([landmarks[end_idx][c] for c in 'xyz'])
    vec = end - start
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else np.array([0, 0, 0])

def calculate_bone_rotations(input_path, output_path):
    setup_logging()
    logging.info(f"Chargement des données lissées depuis: {input_path}")
    try:
        with open(input_path, 'r') as f:
            smoothed_data = json.load(f)
    except Exception as e:
        logging.error(f"Erreur de chargement du JSON: {e}")
        return

    animation_data = {}
    for person_id, track_data in smoothed_data.items():
        logging.info(f"Calcul des rotations pour la personne ID: {person_id}")
        person_anim = []
        for frame_data in track_data:
            if not frame_data['landmarks'] or 'pose' not in frame_data['landmarks']:
                continue

            landmarks = frame_data['landmarks']['pose']
            if not landmarks or len(landmarks) < 33:
                continue

            frame_rotations = {'frame': frame_data['frame'], 'rotations': {}}

            # --- Calcul de l'orientation de la racine (Hips) ---
            hip_right_vec = get_vector(landmarks, SKELETON_BONES['hips'][0], SKELETON_BONES['hips'][1])
            spine_vec = get_vector(landmarks, 24, 12) # Pelvis to shoulder center
            
            # Assurer l'orthogonalité
            hip_forward_vec = np.cross(spine_vec, hip_right_vec)
            hip_forward_vec /= np.linalg.norm(hip_forward_vec)
            hip_up_vec = np.cross(hip_right_vec, hip_forward_vec)

            # Matrice de rotation pour la racine
            root_matrix = np.array([hip_right_vec, hip_up_vec, -hip_forward_vec]).T
            root_rotation = R.from_matrix(root_matrix)
            frame_rotations['rotations']['hips'] = root_rotation.as_quat().tolist() # [x, y, z, w]

            # --- Calcul des rotations des autres os ---
            # Simplification: nous calculons les rotations globales de chaque os.
            # Un vrai retargeting dans un moteur de jeu calculera les rotations locales.
            for bone_name, (start_idx, end_idx) in SKELETON_BONES.items():
                if bone_name == 'hips': continue
                
                bone_vec = get_vector(landmarks, start_idx, end_idx)
                if np.all(bone_vec == 0): continue

                # La rotation est calculée par rapport à la pose de base (Y-up)
                # Ex: un bras pointe vers X+ en T-pose, une jambe vers Y-
                if 'leg' in bone_name: t_pose_vec = np.array([0, -1, 0])
                elif 'arm' in bone_name: t_pose_vec = np.array([1, 0, 0] if 'left' in bone_name else [-1, 0, 0])
                else: t_pose_vec = np.array([0, 1, 0]) # Spine, neck

                # align_vectors retourne la rotation pour passer du premier vecteur au second
                rot, _ = R.align_vectors([bone_vec], [t_pose_vec])
                frame_rotations['rotations'][bone_name] = rot.as_quat().tolist()

            person_anim.append(frame_rotations)
        animation_data[person_id] = person_anim

    logging.info(f"Sauvegarde des données de rotation dans: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(animation_data, f, indent=4)
    logging.info("Calcul des rotations terminé.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calcule les rotations (quaternions) des os à partir de données de landmarks lissées.")
    parser.add_argument("--input_json", required=True, help="Fichier JSON d'entrée (données lissées).")
    parser.add_argument("--output_json", required=True, help="Fichier JSON de sortie pour les rotations.")
    args = parser.parse_args()
    calculate_bone_rotations(args.input_json, args.output_json)

# Exemple:
# python calculate_rotations.py --input_json holistic_motion_smoothed.json --output_json animation_rotations.json
