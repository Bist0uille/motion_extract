import bpy
import json
import sys
import os
import argparse
from mathutils import Quaternion

# Définition du squelette - Doit correspondre à calculate_rotations.py
BONE_HIERARCHY = {
    'hips': None, # Racine
    'spine': 'hips',
    'neck': 'spine',
    'head': 'neck',
    'left_upper_arm': 'spine',
    'left_lower_arm': 'left_upper_arm',
    'right_upper_arm': 'spine',
    'right_lower_arm': 'right_upper_arm',
    'left_upper_leg': 'hips',
    'left_lower_leg': 'left_upper_leg',
    'right_upper_leg': 'hips',
    'right_lower_leg': 'right_upper_leg',
}

# Positions approximatives en T-Pose (en mètres)
BONE_T_POSE_HEADS = {
    'hips': (0, 0.9, 0),
    'spine': (0, 1.0, 0),
    'neck': (0, 1.4, 0),
    'head': (0, 1.55, 0),
    'left_upper_arm': (0.05, 1.4, 0),
    'left_lower_arm': (0.4, 1.4, 0),
    'right_upper_arm': (-0.05, 1.4, 0),
    'right_lower_arm': (-0.4, 1.4, 0),
    'left_upper_leg': (0.1, 0.9, 0),
    'left_lower_leg': (0.1, 0.5, 0),
    'right_upper_leg': (-0.1, 0.9, 0),
    'right_lower_leg': (-0.1, 0.5, 0),
}

BONE_T_POSE_TAILS = {
    'hips': (0, 1.0, 0), # Va vers la colonne
    'spine': (0, 1.4, 0),
    'neck': (0, 1.55, 0),
    'head': (0, 1.7, 0),
    'left_upper_arm': (0.4, 1.4, 0),
    'left_lower_arm': (0.7, 1.4, 0),
    'right_upper_arm': (-0.4, 1.4, 0),
    'right_lower_arm': (-0.7, 1.4, 0),
    'left_upper_leg': (0.1, 0.5, 0),
    'left_lower_leg': (0.1, 0.1, 0),
    'right_upper_leg': (-0.1, 0.5, 0),
    'right_lower_leg': (-0.1, 0.1, 0),
}

def create_armature(name="MotionArmature"):
    bpy.ops.object.add(type='ARMATURE', enter_editmode=True, location=(0, 0, 0))
    armature_obj = bpy.context.object
    armature_obj.name = name
    armature = armature_obj.data
    armature.name = name + "_Data"

    # Créer les os
    for bone_name in BONE_HIERARCHY.keys():
        bone = armature.edit_bones.new(bone_name)
        bone.head = BONE_T_POSE_HEADS[bone_name]
        bone.tail = BONE_T_POSE_TAILS[bone_name]

    # Définir la hiérarchie
    for bone_name, parent_name in BONE_HIERARCHY.items():
        if parent_name:
            armature.edit_bones[bone_name].parent = armature.edit_bones[parent_name]

    bpy.ops.object.mode_set(mode='OBJECT')
    return armature_obj

def apply_animation(armature_obj, animation_data):
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    for frame_info in animation_data:
        frame_num = frame_info['frame']
        bpy.context.scene.frame_set(frame_num)

        for bone_name, quat_xyzw in frame_info['rotations'].items():
            if bone_name in armature_obj.pose.bones:
                pose_bone = armature_obj.pose.bones[bone_name]
                pose_bone.rotation_mode = 'QUATERNION'
                # Conversion de Scipy (x, y, z, w) à Blender (w, x, y, z)
                quat_wxyz = Quaternion((quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]))
                pose_bone.rotation_quaternion = quat_wxyz
                pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_num)

    bpy.ops.object.mode_set(mode='OBJECT')

def export_fbx(armature_obj, output_path):
    bpy.ops.object.select_all(action='DESELECT')
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj

    # Paramètres d'export pour Unreal Engine
    bpy.ops.export_scene.fbx(
        filepath=output_path,
        use_selection=True,
        axis_forward='-Y', # L'avant dans Blender devient l'avant dans UE
        axis_up='Z',      # Le haut dans Blender devient le haut dans UE
        object_types={'ARMATURE'},
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_nla_strips=False, # Simplifie l'export d'animation
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=False,
    )

def main():
    # --- Parsing des arguments spécifiques à Blender ---
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Script Blender pour exporter une animation en FBX.")
    parser.add_argument("--input_json", required=True, help="Fichier JSON d'entrée (rotations).")
    parser.add_argument("--output_fbx", required=True, help="Fichier FBX de sortie.")
    args = parser.parse_args(argv)

    # --- Exécution ---
    print(f"[Blender] Chargement de: {args.input_json}")
    with open(args.input_json, 'r') as f:
        animation_data = json.load(f)
    
    # On ne traite que la première personne
    person_id = next(iter(animation_data))
    person_anim_data = animation_data[person_id]

    # Nettoyer la scène
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    armature = create_armature()
    apply_animation(armature, person_anim_data)
    export_fbx(armature, args.output_fbx)

    print(f"[Blender] Exportation terminée vers: {args.output_fbx}")

if __name__ == "__main__":
    main()
