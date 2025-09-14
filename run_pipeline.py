import argparse
import subprocess
import os
import logging
import sys

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def run_command(command):
    """Exécute une commande en affichant sa sortie en temps réel."""
    logging.info(f"Lancement de: {' '.join(command)}")
    
    # En ne passant pas stdout/stderr, ils sont hérités du parent et s'affichent directement.
    result = subprocess.run(command)
    
    if result.returncode != 0:
        logging.error(f"L'étape précédente a échoué avec le code {result.returncode}. Arrêt du pipeline.")
        sys.exit(1)
    
    logging.info("Étape terminée avec succès.")

def main():
    parser = argparse.ArgumentParser(description="Pipeline complet pour l'extraction d'animation 3D.")
    parser.add_argument("--input_video", required=True, help="Vidéo d'entrée.")
    parser.add_argument("--output_dir", default="output", help="Dossier de sortie.")
    parser.add_argument("--start_time", help="Début (MM:SS).")
    parser.add_argument("--end_time", help="Fin (MM:SS).")
    parser.add_argument("--conf", type=float, default=0.5, help="Seuil de confiance YOLO.")
    
    # Contrôle du pipeline
    parser.add_argument("--skip_extraction", action="store_true")
    parser.add_argument("--skip_smoothing", action="store_true")
    parser.add_argument("--skip_rotation", action="store_true")
    parser.add_argument("--no_visualization", action="store_true")
    parser.add_argument("--export_fbx", action="store_true", help="Active l'exportation finale en FBX via Blender.")
    parser.add_argument("--blender_path", help="Chemin vers l'exécutable de Blender.")

    args = parser.parse_args()
    setup_logging()

    if args.export_fbx and not args.blender_path:
        logging.error("Le chemin vers Blender (--blender_path) est requis pour l'exportation FBX.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(args.input_video))[0]
    raw_json_path = os.path.join(args.output_dir, f"{base_name}_raw.json")
    smoothed_json_path = os.path.join(args.output_dir, f"{base_name}_smoothed.json")
    rotations_json_path = os.path.join(args.output_dir, f"{base_name}_rotations.json")
    fbx_path = os.path.join(args.output_dir, f"{base_name}_animation.fbx")
    python_executable = sys.executable

    if not args.skip_extraction:
        cmd = [python_executable, "extract_motion.py", "--input_video", args.input_video, "--output_json", raw_json_path, "--conf", str(args.conf)]
        if args.start_time: cmd.extend(["--start_time", args.start_time])
        if args.end_time: cmd.extend(["--end_time", args.end_time])
        run_command(cmd)

    if not args.skip_smoothing:
        run_command([python_executable, "smoother.py", "--input_json", raw_json_path, "--output_json", smoothed_json_path])

    if not args.skip_rotation:
        run_command([python_executable, "calculate_rotations.py", "--input_json", smoothed_json_path, "--output_json", rotations_json_path])

    if args.export_fbx:
        logging.info("Lancement de l'exportation FBX avec Blender...")
        cmd = [
            args.blender_path, "--background", "--python", "export_to_fbx.py", "--",
            "--input_json", rotations_json_path,
            "--output_fbx", fbx_path
        ]
        run_command(cmd)

    if not args.no_visualization:
        logging.info("Lancement de la visualisation 3D...")
        run_command([python_executable, "visualize_animation.py", "--input_json", smoothed_json_path])

    logging.info("Pipeline terminé.")

if __name__ == '__main__':
    main()
