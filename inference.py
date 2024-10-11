from ast import arg
import os
import sys

PROJECT_PATH = os.path.abspath('.')
sys.path.insert(0, PROJECT_PATH)
PYTHON_PATH = os.popen('which python').read().strip()
if sys.executable.endswith('blender'):
    INSIDE_BLENDER = True
    sys.executable = PYTHON_PATH
else:
    INSIDE_BLENDER = False

import json
import argparse
from model.inference import InferenceTool, InferenceCode
from utils.utils import extract_args

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input_instruction_json', default=''
)
parser.add_argument(
    '--input_scene_dir', default=''
)
parser.add_argument(
    '--instruction_idx', default=0, type=int
)

def main(args):
    assert args.input_scene_dir != ''
    assert args.input_instruction_json != ''
    instruction_model_params = {
        "name": "GTLoader"
    }
    visual_recognition_model_params = {
        "name": "GTLoader"
    }
    pose_model_params = {
        "name": "GTLoader"
    }
    action_planner_params = {
        "name": "GTLoader"
    }

    if not INSIDE_BLENDER:
        environment_params = {
            "robots":'Panda',
            "gripper_types":'Robotiq85GripperContinuous',
            "has_offscreen_renderer": False,
            "ignore_done": True,
            "use_camera_obs": False,
            "use_object_obs": True,
            "control_freq": 20,
            "render_collision_mesh": True,
            "renderer":'mujoco',
            "renderer_config":None,
            "bounding_boxes_from_scene":True,
            "debug":True,
            "has_renderer": True,
            "return_bboxes": True,
            "list_obj_properties": True,
            "blender_render": False,
            "output_render_path": f'test_images/temp',
            "controller_config_path": "./environment/controller_cfg_test.json"
        }
    else:
        environment_params = {
            "robots":'Panda',
            "gripper_types":'Robotiq85GripperContinuous',
            "has_offscreen_renderer": False,
            "ignore_done": True,
            "use_camera_obs": False,
            "use_object_obs": True,
            "control_freq": 20,
            "render_collision_mesh": True,
            "renderer":'mujoco',
            "renderer_config":None,
            "bounding_boxes_from_scene":True,
            "return_bboxes":True,
            "debug":True,
            "list_obj_properties":True,
            "has_renderer": False,
            "blender_render": True,
            "output_render_path": f'test_images/inference_test/temp2',
            "blender_config_path": "./environment/blender_default_cfg.json",
            "controller_config_path": "./environment/controller_cfg_test.json"
        }

    with open(args.input_instruction_json, 'r') as f:
        instruction_struct = json.load(f)

    sequences_dir = os.path.join(args.input_scene_dir, '..', 'sequences')
    os.makedirs(sequences_dir, exist_ok=True)

    instr = instruction_struct['instructions'][args.instruction_idx]

    base_fname = os.path.splitext(instr['image_filename'])[0]
    scene_json = os.path.join(args.input_scene_dir, base_fname + '.json')

    save_dir = os.path.join(sequences_dir, base_fname)
    save_path = os.path.join(save_dir, 'frame')
    os.makedirs(save_dir, exist_ok=True)

    environment_params['output_render_path'] = save_path

    with open(scene_json, 'r') as f:
        scene = json.load(f)

    inference_tool = InferenceTool()
    inference_tool.setup(
        instruction_model_params=instruction_model_params,
        visual_recognition_model_params=visual_recognition_model_params,
        environment_params=environment_params,
        pose_model_params=pose_model_params,
        action_planner_params=action_planner_params,
        disable_rendering=False,
        save_dir=save_dir
    )

    inference_tool.load_scene_gt(scene)
    inference_tool.load_instruction(instr)

    outcome = inference_tool.run()
    print(InferenceCode.code_to_str(outcome))
    inference_tool.outcome_to_json(outcome)

if __name__ == '__main__':
    if INSIDE_BLENDER:
        argv = extract_args()
        args = parser.parse_args(argv)
        main(args)
    else:
        args = parser.parse_args()
        print(args)
        main(args)
