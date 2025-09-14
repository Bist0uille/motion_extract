
import cv2
import mediapipe as mp
import json
import argparse
import os
import logging
from ultralytics import YOLO
from mediapipe.framework.formats import landmark_pb2

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def parse_time_to_seconds(time_str):
    if not time_str: return 0
    parts = list(map(int, time_str.split(':')))
    if len(parts) == 2: return parts[0] * 60 + parts[1]
    if len(parts) == 3: return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0

def landmarks_to_dict(landmark_list):
    """Convertit une liste de landmarks MediaPipe en liste de dictionnaires."""
    if not landmark_list: return None
    return [{'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility} for lm in landmark_list.landmark]

def translate_landmarks_for_drawing(landmarks, crop_w, crop_h, x_offset, y_offset, frame_w, frame_h):
    """Traduit les coordonnées normalisées d'un crop vers l'image complète."""
    translated_landmarks = landmark_pb2.NormalizedLandmarkList()
    for lm in landmarks.landmark:
        new_lm = translated_landmarks.landmark.add()
        new_lm.x = (lm.x * crop_w + x_offset) / frame_w
        new_lm.y = (lm.y * crop_h + y_offset) / frame_h
        new_lm.z = lm.z
        new_lm.visibility = lm.visibility
    return translated_landmarks

def extract_holistic_motion(video_path, output_path, start_time_str=None, end_time_str=None, preview=False, conf=0.4, tracker="bytetrack.yaml"):
    setup_logging()
    logging.info(f"Initialisation des modèles. Tracker: {tracker}, Confiance: {conf}.")
    yolo_model = YOLO('yolov8n.pt')
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    holistic = mp_holistic.Holistic(
        static_image_mode=False, model_complexity=2, # Upping complexity for better results
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    logging.info("Modèles chargés.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Impossible d'ouvrir la vidéo: {video_path}")
        return

    fps, total_frames, frame_h, frame_w = (cap.get(p) for p in [cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_WIDTH])
    frame_h, frame_w = int(frame_h), int(frame_w)

    start_frame = int(parse_time_to_seconds(start_time_str) * fps) if start_time_str else 0
    end_frame = int(parse_time_to_seconds(end_time_str) * fps) if end_time_str else total_frames
    if start_frame > 0: cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    logging.info(f"Traitement de l'image {start_frame} à {end_frame}.")

    tracked_skeletons = {}
    frame_count = start_frame

    while cap.isOpened():
        if frame_count >= end_frame: break
        success, frame = cap.read()
        if not success: break
        frame_count += 1

        annotated_frame = frame.copy() if preview else None
        yolo_results = yolo_model.track(source=frame, tracker=tracker, classes=0, conf=conf, verbose=False)
        
        if yolo_results[0].boxes.id is not None:
            tracked_boxes = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = yolo_results[0].boxes.id.cpu().numpy().astype(int)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            for box, track_id in zip(tracked_boxes, track_ids):
                x1, y1, x2, y2 = box
                padding = 10
                y1_pad, y2_pad = max(0, y1 - padding), min(frame_h, y2 + padding)
                x1_pad, x2_pad = max(0, x1 - padding), min(frame_w, x2 + padding)
                
                person_crop = image_rgb[y1_pad:y2_pad, x1_pad:x2_pad]
                if person_crop.size == 0: continue

                results = holistic.process(person_crop)

                if results.pose_world_landmarks:
                    all_landmarks = {
                        'pose': landmarks_to_dict(results.pose_world_landmarks),
                        'face': landmarks_to_dict(results.face_landmarks),
                        'left_hand': landmarks_to_dict(results.left_hand_landmarks),
                        'right_hand': landmarks_to_dict(results.right_hand_landmarks)
                    }
                    if track_id not in tracked_skeletons: tracked_skeletons[track_id] = []
                    tracked_skeletons[track_id].append({'frame': frame_count, 'landmarks': all_landmarks})

                if preview:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    crop_h, crop_w, _ = person_crop.shape
                    
                    # Dessin du squelette complet
                    for landmark_type, connections, color in [
                        (results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, (255,0,0)),
                        (results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, (80,110,10)),
                        (results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, (80,22,10)),
                        (results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, (80,44,121))]:
                        
                        if landmark_type:
                            translated_lm = translate_landmarks_for_drawing(landmark_type, crop_w, crop_h, x1_pad, y1_pad, frame_w, frame_h)
                            mp_drawing.draw_landmarks(annotated_frame, translated_lm, connections, 
                                                    mp_drawing.DrawingSpec(color=color, thickness=1, circle_radius=1),
                                                    mp_drawing.DrawingSpec(color=color, thickness=1))

            if preview:
                cv2.imshow("MotionExtract Preview", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

        if frame_count % 30 == 0: logging.info(f"Image traitée: {frame_count}/{int(end_frame)}")

    cap.release()
    holistic.close()
    if preview: cv2.destroyAllWindows()
    logging.info("Ressources libérées.")

    with open(output_path, 'w') as f: json.dump({str(k): v for k, v in tracked_skeletons.items()}, f, indent=4)
    logging.info(f"Données pour {len(tracked_skeletons)} pistes sauvegardées dans: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extrait l'animation 3D complète (corps, visage, mains) de plusieurs personnes.")
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--start_time", default=None, help="Début (MM:SS).")
    parser.add_argument("--end_time", default=None, help="Fin (MM:SS).")
    parser.add_argument("--preview", action="store_true", help="Affiche une prévisualisation.")
    parser.add_argument("--conf", type=float, default=0.4, help="Seuil de confiance YOLO (0.0-1.0).")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Tracker à utiliser (ex: bytetrack.yaml).")
    args = parser.parse_args()
    extract_holistic_motion(args.input_video, args.output_json, args.start_time, args.end_time, args.preview, args.conf, args.tracker)
