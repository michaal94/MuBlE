import os
import json
import copy
import numpy as np

from PIL import Image

from robosuite import load_controller_config
import robosuite.utils.transform_utils as T

from model import INSTRUCTION_MODELS, VISUAL_RECOGNITION_MODELS
from model import POSE_MODELS, ACTION_PLAN_MODELS

from environment.scene_parser import SceneParser
from environment.tabletop_env import TabletopEnv
from environment.actions import ActionExecutor

from model.program_executor import ProgramStatus, ProgramExecutor

from utils.utils import CyclicBuffer

class InferenceCode:
    CORRECT_ANSWER = 0
    SUCCESSFUL_TASK = 1
    PROGRAM_FAILURE = 2
    INCORRECT_ANSWER = 3
    TIMEOUT = 4
    TASK_FAILURE = 5
    EXECUTION_ERROR = 6
    LOOP_ERROR = 7
    BROKEN_SCENE = 8

    codebook = [
        "CORRECT_ANSWER",
        "SUCCESSFUL_TASK",
        "PROGRAM_FAILURE",
        "INCORRECT_ANSWER",
        "TIMEOUT",
        "TASK_FAILURE",
        "EXECUTION_ERROR",
        "LOOP_ERROR",
        "BROKEN_SCENE"
    ]

    @staticmethod
    def code_to_str(code):
        return InferenceCode.codebook[code]

class InferenceTool:
    def __init__(self) -> None:
        self.pose_model = None
        self.instruction_model = None
        self.visual_recognition_model = None
        self.scene_gt = None
        self.instruction = None
        # self.scene = None
        self.blender_rendering = False

    def setup(self,
              instruction_model_params={},
              visual_recognition_model_params={},
              pose_model_params={},
              action_planner_params={},
              environment_params={},
              program_executor_params={},
              scene_parser_params={},
              action_executor_params={},
              timeout = 10,
              planning_timeout = 20,
              env_timeout = 1000,
              disable_rendering = False, 
              save_dir = 'temp',
              verbose = True
        ):
        self.verbose = verbose
        self.save_dir = save_dir
        self.disable_rendering = disable_rendering
        self.timeout = timeout
        self.env_timeout = env_timeout
        self.planning_timeout = planning_timeout
        self.environment_params = environment_params
        self._setup_instruction_model(instruction_model_params)
        self._setup_visual_recognition_model(visual_recognition_model_params)
        self._setup_pose_model(pose_model_params)
        self._setup_action_planner(action_planner_params)
        self._setup_visual_recognition_model_gt()
        self._setup_pose_model_gt()
        self._setup_action_planner_gt()
        self.scene_parser = SceneParser(**scene_parser_params)
        self.program_executor = ProgramExecutor(**program_executor_params)
        self.action_executor = ActionExecutor(**action_executor_params)
        self.obs_num = 0
        self.last_render_path = None
        self.loop_detector = CyclicBuffer(2)

    def run(self):
        assert self.instruction_model is not None, "Load instruction to program model (or set GT instruction mode) before running inference"
        assert self.visual_recognition_model is not None, "Load visual recognition model"
        assert self.pose_model is not None, "Load pose estimation model (or set GT pose mode) before running inference"


        # Get program from instruction
        program_list = self.instruction_model.get_program(self.instruction)
        if self.verbose:
            print(f'Program:\t{program_list}')

        # Setup environment
        self._setup_environment()
        # Run some iteration to apply gravity to objects
        action = self.default_environment_action
        _, _, _, _ = self.environment.step(action)
        observation, _, _, _ = self.environment.step(action)
        # input()
        #DEBUG
        self.action_executor.env = self.environment

        # Get first image
        if self.environment.blender_enabled:
            image_path = self.environment.blender_render()
            self.last_render_path = image_path
            image = self._load_image(image_path)
        else:
            # Without blender we don't use the environment to output images
            # Rather for debugging
            # Can be changed though (save images from MuJoCo)
            if not self.disable_rendering:
                self.environment.render()
            image = None

        poses, bboxes = self.pose_model.get_pose(image, observation)
        poses_gt, bboxes_gt = self.pose_model_gt.get_pose(image, observation)
        
        scene_vis = self.visual_recognition_model.get_scene(image, None, self.scene_gt)
        scene_vis_gt = self.visual_recognition_model_gt.get_scene(image, None, self.scene_gt)
        
        self.scene_graph = self._make_scene_graph(scene_vis, poses, bboxes)
        self.scene_graph_gt = self._make_scene_graph(scene_vis_gt, poses_gt, bboxes_gt)


        if self.environment.blender_enabled:
            self.update_sequence_json(observation, ('START', None))

        if not self._check_gt_scene():
            print("Broken scene error")
            return InferenceCode.BROKEN_SCENE

        self.task = self._get_task_from_instruction()
        # print(self.scene_graph)
        # exit()

        for _ in range(self.timeout):
            # self.scene_graph[0]['weight'] = np.array(160)
            # self.scene_graph[1]['weight'] = np.array(140)
            # self.scene_graph[2]['weight'] = np.array(113.6)
            # self.scene_graph[3]['weight'] = np.array(163.4)
            program_output = self.program_executor.execute(self.scene_graph, program_list)
            self.loop_detector.flush()

            print(program_output)
            # print(sce)
            # exit()
            program_status = program_output['STATUS']

            if program_status == ProgramStatus.FAILURE:
                print('Program ended with failure')
                return InferenceCode.PROGRAM_FAILURE

            if program_status == ProgramStatus.SUCCESS:
                print('Program passed through scene graph')
                if self._check_answer(program_output['ANSWER']):
                    print('Answer correct')
                    return InferenceCode.CORRECT_ANSWER
                else:
                    print('Answer incorrect')
                    return InferenceCode.INCORRECT_ANSWER

            if program_status == ProgramStatus.ACTION or program_status == ProgramStatus.FINAL_ACTION:
                planning_tout = self.planning_timeout * len(program_output['ACTION']['target'])
                for _ in range(planning_tout):
                    # print(self.scene_graph_gt[2]['in_hand'], self.scene_graph_gt[2]['raised'])
                    # print(self.scene_graph[2]['in_hand'], self.scene_graph[2]['raised'])
                    # self.scene_graph[2]['in_hand'] = True
                    # self.scene_graph[2]['raised'] = True
                    action_plan = self.action_planner.get_action_sequence(
                        program_output['ACTION'],
                        self.scene_graph 
                    )
                    # print(self.scene_graph[1]['in_hand'])
                    # print(self.scene_graph[1]['raised'])
                    # print(self.scene_graph[1]['approached'])
                    # print(self.scene_graph[1]['gripper_over'])
                    # print(self.scene_graph[1]['pos'])
                    # print(self.scene_graph[1]['bbox'])
                    # print(self.scene_graph[2]['in_hand'])
                    # print(self.scene_graph[2]['raised'])
                    # print(self.scene_graph[2]['approached'])
                    # print(self.scene_graph[2]['gripper_over'])
                    # print(self.scene_graph[2]['pos'])
                    # print(self.scene_graph[2]['bbox'])
                    print(action_plan)
                    if self._detect_loop(action_plan):
                        print('Loop detected, exiting')
                        return InferenceCode.LOOP_ERROR
                    self.loop_detector.append(action_plan)
                    # input()
                    # exit()
                    # input()
                    if len(action_plan) == 0:
                        if program_status == ProgramStatus.ACTION:
                            break
                        if self._check_task_completion(self._get_task_from_instruction(), observation):
                            if program_status == ProgramStatus.FINAL_ACTION:
                                print('Correct execution')
                                return InferenceCode.SUCCESSFUL_TASK
                            else:
                                break
                        else:
                            print('Task not reached target')
                            return InferenceCode.TASK_FAILURE
                    action_to_execute = action_plan[0]
                    self.action_executor.set_action(
                        action_to_execute[0], 
                        action_to_execute[1],
                        observation,
                        self.scene_graph
                    )
                    action_executed = False
                    for _ in range(self.env_timeout):
                        if self.action_executor.get_current_action():
                            action = self.action_executor.step(observation)
                        else:
                            action_executed = True
                            break
                        observation, _, _, _ = self.environment.step(action)
                        # input()
                        if not self.environment.blender_enabled:
                            if not self.disable_rendering:
                                self.environment.render()
                    if action_executed:
                        if self.environment.blender_enabled:
                            image_path = self.environment.blender_render()
                            self.last_render_path = image_path
                            image = self._load_image(image_path)
                        poses, bboxes = self.pose_model.get_pose(image, observation)
                        poses_gt, bboxes_gt = self.pose_model_gt.get_pose(image, observation)
                        self._update_scene_graph(poses, bboxes, observation)
                        self._update_scene_graph(poses_gt, bboxes_gt, observation, gt=True)
                        if not self._check_gt_scene():
                            print("Broken scene error")
                            return InferenceCode.BROKEN_SCENE
                        if self.environment.blender_enabled:
                            self.update_sequence_json(observation, action_to_execute)
                    else:
                        print('Action not executed correctly')
                        return InferenceCode.EXECUTION_ERROR


        print("Timeout, program execution reiterations exceeded.")
        return InferenceCode.TIMEOUT

        # assert self.pose_model is not None, "Load pose estimation model (or set GT pose mode) before running inference"

        # default_action = self._get_default_action()
        # for i in range(100):
        #     self.environment.step(default_action)
        #     if self.blender_rendering:
        #         self.environment.blender_render()
        #     else:
        #         self.environment.render()

        '''
        + Get program from instruction
        + Setup environment
        + Get first image
        + Get first poses
        + Get visual recognition
        + Associate semantic and geometric graphs
        while before timeout:
            Get program output on the scene graph
            if failure:
                exit
            if success:
                check answer and exit
            if action or action_final:
                for all tasks: 
                    Loop:       
                        if no action happening:
                            Check last reward - check if success 
                            Check last observations (eef pos and ori)
                            Get image
                            Get poses
                            align new poses with old ones (assign to proper graph nodes)
                            update_scene_graph
                            get primitive list
                            set current action in actionexecutor to primitives[0]
                        else:
                            action.step()
                        env.step()
                if action_final:
                    check last status and exit
        '''

    # def _load_scene(self, scene):
    #     assert 'objects' in scene
    #     self.scene = scene

    def _update_scene_graph(self, poses, bboxes, obs, gt=False):
        if gt:
            for i in range(len(self.scene_graph_gt)):
                self.scene_graph_gt[i]['pos'] = poses[i][0]
                self.scene_graph_gt[i]['ori'] = poses[i][1]
                self.scene_graph_gt[i]['bbox'] = bboxes[i]
            eef_pos = obs['robot0_eef_pos']
            eef_ori = obs['robot0_eef_quat']
            gripper_closed = obs['gripper_closed']
            gripper_action = obs['gripper_action']
            # Iterate again cause ultimately there will be matching 
            for idx, obj in enumerate(self.scene_graph_gt):
                obj['in_hand'] = False
                obj['raised'] = False
                obj['approached'] = False
                obj['gripper_over'] = False
                bbox_boundaries = self._check_eef_in_bbox(obj, eef_pos)
                finger_in_bbox = self._check_finger_in_bbox(obj, eef_pos, eef_ori)
                # print(obj['name'], finger_in_bbox)
                # print(bbox_boundaries)
                # print(gripper_closed)
                # print(gripper_action)
                if len(bbox_boundaries) == 3 or finger_in_bbox:
                    if gripper_closed and gripper_action > -0.99:
                        obj['in_hand'] = True
                    else:
                        obj['approached'] = True
                elif ('x' in bbox_boundaries and 'y' in bbox_boundaries):
                    obj['gripper_over'] = True
                if all([obj['bbox'][i][2] > 0.05 for i in range(8)]):
                    obj['raised'] = True
                    if obs['grasped_obj_idx'] == idx:
                        obj['weight'] = obs['weight_measurement']
        else:
            #TODO
            # Do the pose matching w.r.t history
            for i in range(len(self.scene_graph)):
                self.scene_graph[i]['pos'] = poses[i][0]
                self.scene_graph[i]['ori'] = poses[i][1]
                self.scene_graph[i]['bbox'] = bboxes[i]
            eef_pos = obs['robot0_eef_pos']
            eef_ori = obs['robot0_eef_quat']
            gripper_closed = obs['gripper_closed']
            gripper_action = obs['gripper_action']
            # print(obs)
            # Iterate again cause ultimately there will be matching 
            for obj in self.scene_graph:
                obj['in_hand'] = False
                obj['raised'] = False
                obj['approached'] = False
                obj['gripper_over'] = False
                bbox_boundaries = self._check_eef_in_bbox(obj, eef_pos)
                finger_in_bbox = self._check_finger_in_bbox(obj, eef_pos, eef_ori)
                # print(bbox_boundaries)
                # print(gripper_closed)
                # print(gripper_action)
                if len(bbox_boundaries) == 3 or finger_in_bbox:
                    if gripper_closed and gripper_action > -0.99:
                        obj['in_hand'] = True
                    else:
                        obj['approached'] = True
                elif ('x' in bbox_boundaries and 'y' in bbox_boundaries):
                    obj['gripper_over'] = True
                if all([obj['bbox'][i][2] > 0.05 for i in range(8)]):
                    obj['raised'] = True
                if obj['raised'] and obj['in_hand']:
                    obj['weight'] = obs['weight_measurement']
                # print(obj['name'], obj['in_hand'], obj['raised'])

    def _check_eef_in_bbox(self, obj, eef_pos):
        bbox = obj['bbox']
        x_vec = bbox[0] - bbox[1]
        y_vec = bbox[0] - bbox[3]
        z_vec = bbox[4] - bbox[0]
        x_size = np.linalg.norm(x_vec)
        y_size = np.linalg.norm(y_vec)
        z_size = np.linalg.norm(z_vec)
        # print(bbox.shape)
        bbox_mid = np.mean(bbox, axis=0)
        mid_to_eef_vec = eef_pos - bbox_mid
        axes_in = []
        x_proj = np.abs(np.dot(mid_to_eef_vec, x_vec) / x_size)
        if x_proj < x_size / 2:
            axes_in.append('x')
        y_proj = np.abs(np.dot(mid_to_eef_vec, y_vec) / y_size)
        if y_proj < y_size / 2:
            axes_in.append('y')
        z_proj = np.abs(np.dot(mid_to_eef_vec, z_vec) / z_size)
        if z_proj < z_size / 2:
            axes_in.append('z')
        return axes_in 

    def _check_finger_in_bbox(self, obj, eef_pos, eef_ori):
        finger_dist = self.action_executor.gripper_open_width
        gripper_y_dir = T.quat2mat(eef_ori)[:, 1]
        gripper_y_dir = gripper_y_dir / np.linalg.norm(gripper_y_dir)
        finger1 = eef_pos + finger_dist * gripper_y_dir
        finger2 = eef_pos - finger_dist * gripper_y_dir
        if len(self._check_eef_in_bbox(obj, finger1)) == 3:
            return True
        if len(self._check_eef_in_bbox(obj, finger2)) == 3:
            return True
        return False


    def _check_task_completion(self, action, obs):
        if action['task'] == 'measure_weight':
            if self.scene_graph_gt[action['target'][0][0]]['raised']:
                if obs['weight_measurement'] > 0:
                    self.scene_graph_gt[action['target'][0][0]]['weight'] = obs['weight_measurement']
                    return True
        elif action['task'] == 'pick_up':
            print(action['target'][0][0])
            if self.scene_graph_gt[action['target'][0][0]]['raised']:
                 return True
        elif action['task'] == 'stack':
            return self._check_stack(action['target'][0])
        elif 'move' in action['task']:
            return self._check_move(action)
        return False

    def _check_move(self, task):
        side = task['task'].split('[')[1].strip(']')
        for idx in task['target'][0]:
            if self.scene_graph_gt[idx]['in_hand']:
                return False
            y_pos = self.scene_graph_gt[idx]['pos'][1]
            if (side == 'right' and y_pos < 0):
                return False
            if (side == 'left' and y_pos > 0):
                return False
        return True

    def _check_stack(self, targets):
        targets = targets[0]
        for i in reversed(range(len(targets) - 1)):
            top = targets[i]
            bottom = targets[i + 1]
            pos_top = self.scene_graph_gt[top]['pos']
            pos_bot = self.scene_graph_gt[bottom]['pos']
            if (pos_top[2] - pos_bot[2]) < 0.0:
                return False
            if np.linalg.norm(pos_top[0:2] - pos_bot[0:2]) > 0.1:
                return False
        return True

    def _check_answer(self, answer):
        gt_answer = self.instruction['program'][-1]['output']
        if gt_answer is None:
            return False
        if answer is None:
            return False
        # print(answer)
        if isinstance(answer[0], list):
            if not isinstance(gt_answer[0], list):
                # answer = [a[0] for a in answer]
                answer = answer[0]
                # print(answer)
                if type(answer[0]).__module__ == np.__name__:
                    answer = [a.tolist() for a in answer]
        # print(gt_answer, answer)
        
        if len(answer) != len(gt_answer):
            return False
        for a in answer:
            if a not in gt_answer:
                return False
        return True

    def _make_scene_graph(self, scene, poses, bboxes):
        scene_graph = []
        for i, o in enumerate(scene):
            scene_graph.append(copy.deepcopy(o))
            scene_graph[-1]['pos'] = poses[i][0]
            scene_graph[-1]['ori'] = poses[i][1]
            scene_graph[-1]['bbox'] = bboxes[i]
            scene_graph[-1]['in_hand'] = False
            scene_graph[-1]['raised'] = False
            scene_graph[-1]['approached'] = False
            scene_graph[-1]['gripper_over'] = False
        return scene_graph

    def load_scene_gt(self, scene):
        assert 'objects' in scene
        self.scene_gt = scene
    
    def load_instruction(self, instruction_dict):
        assert 'instruction' in instruction_dict
        self.instruction = instruction_dict

    def _setup_instruction_model(self, params):
        assert params['name'] in INSTRUCTION_MODELS, "Unknown instruction model"
        self.instruction_model = INSTRUCTION_MODELS[params['name']](params)

    def _setup_visual_recognition_model(self, params):
        assert params['name'] in VISUAL_RECOGNITION_MODELS, "Unknown visual model"
        self.visual_recognition_model = VISUAL_RECOGNITION_MODELS[params['name']](params)

    def _setup_pose_model(self, params):
        assert params['name'] in POSE_MODELS, "Unknown pose model"
        self.pose_model = POSE_MODELS[params['name']](params)

    def _setup_action_planner(self, params):
        assert params['name'] in ACTION_PLAN_MODELS, "Unknown action planner"
        self.action_planner = ACTION_PLAN_MODELS[params['name']](params)

    def _setup_visual_recognition_model_gt(self):
        self.visual_recognition_model_gt = VISUAL_RECOGNITION_MODELS["GTLoader"]()

    def _setup_pose_model_gt(self):
        self.pose_model_gt = POSE_MODELS["GTLoader"]()

    def _setup_action_planner_gt(self):
        self.action_planner_gt = ACTION_PLAN_MODELS["GTLoader"]()

    def _setup_environment(self):
        assert self.scene_gt is not None, "Load GT scene to initialise simulation"
        controller_cfg_path = self.environment_params.pop('controller_config_path', None)
        if controller_cfg_path is not None:
            controller_cfg = load_controller_config(controller_cfg_path)
        else:
            controller_cfg = None
        self.environment_params['controller_configs'] = controller_cfg
        blender_cfg_path = self.environment_params.pop('blender_config_path', None)
        if blender_cfg_path is not None:
            with open(blender_cfg_path, 'r') as f:
                blender_cfg = json.load(f)
        else:
            blender_cfg = None
        self.environment_params['blender_config'] = blender_cfg
        self.environment_params['scene_dict'] = self.scene_gt
        obj_dict = self.scene_parser.parse_scene(self.scene_gt)
        self.environment_params['objs'] = obj_dict

        self.environment = TabletopEnv(**self.environment_params)
        if 'blender_render' in self.environment_params:
            self.blender_rendering = self.environment_params['blender_render']
        else:
            self.blender_rendering = False

        action_dim = self.environment.action_dim
        self.default_environment_action = np.array(
            (action_dim - 1) * [0] + [-1]
        )

    def _load_image(self, path):
        return Image.open(path)

    def _get_task_from_instruction(self):
        prog_last = self.instruction["program"][-1]
        outp = prog_last["output"]
        if isinstance(prog_last["output"][0], list):
            new_outp = []
            for o_item in prog_last["output"]:
                new_outp += o_item
            outp = [new_outp]
        if 'move' in prog_last["type"]:
            out_task = f"{prog_last['type']}[{prog_last['input_value']}]"
        else:
            out_task = prog_last["type"]
        task = {
            'task': out_task,
            'target': [outp]
        }
        return task

    def update_sequence_json(self, obs, last_action=None):
        if not self.environment.blender_enabled:
            return
        if self.obs_num == 0:
            json_name = "sequence.json"
            self.json_path = os.path.join(self.save_dir, json_name)
            info_struct = {
                "info": {
                    "image_filename": self.instruction["image_filename"],
                    "instruction": self.instruction["instruction"],
                    "task": self._get_task_from_instruction()
                },
                "observations": [],
                "observations_gt": [],
                "image_paths": [],
                "result": None
            }
            with open(self.json_path, 'w') as f:
                json.dump(info_struct, f, indent=4)

        with open(self.json_path, 'r') as f:
            info_struct = json.load(f)

        info_struct['image_paths'].append(self.last_render_path)
        obs_set = {}
        obs_set['robot'] = {
            'pos': obs['robot0_eef_pos'].tolist(),
            'ori': obs['robot0_eef_quat'].tolist(),
            'gripper_action': obs['gripper_action'].tolist(),
            'gripper_closed': obs['gripper_closed'].tolist(),
            'weight_measurement': obs['weight_measurement'].tolist(),
            'action': last_action
        }

        obs_set['objects'] = []
        for obj in self.scene_graph:
            obs_set['objects'].append(copy.deepcopy(obj))
            obs_set['objects'][-1]['bbox'] = obs_set['objects'][-1]['bbox'].tolist()
            obs_set['objects'][-1]['pos'] = obs_set['objects'][-1]['pos'].tolist()
            obs_set['objects'][-1]['ori'] = obs_set['objects'][-1]['ori'].tolist()
            if obs_set['objects'][-1]['weight'] is not None:
                obs_set['objects'][-1]['weight'] = obs_set['objects'][-1]['weight'].tolist()

        obs_set_gt = {}
        robot_body, gripper_body = self.environment.get_robot_configuration()
        for k, v in robot_body.items():
            robot_body[k] = (v[0].tolist(), v[1].tolist())
        for k, v in gripper_body.items():
            gripper_body[k] = (v[0].tolist(), v[1].tolist())

        masks = self.environment.get_segmentation_masks()
        obs_set_gt['robot'] = {
            'pos': obs['robot0_eef_pos'].tolist(),
            'ori': obs['robot0_eef_quat'].tolist(),
            'gripper_action': obs['gripper_action'].tolist(),
            'gripper_closed': obs['gripper_closed'].tolist(),
            'weight_measurement': obs['weight_measurement'].tolist(),
            'grasped_obj_idx': obs['grasped_obj_idx'].tolist(),
            'robot_body': robot_body,
            'gripper_body': gripper_body,
            'robot_mask': masks['robot'],
            'table_mask': masks['table']
        }

        obs_set_gt['objects'] = []
        for i, obj in enumerate(self.scene_graph_gt):
            obs_set_gt['objects'].append(copy.deepcopy(obj))
            obs_set_gt['objects'][i]['mask'] = masks['objects'][i]
            obs_set_gt['objects'][-1]['bbox'] = obs_set_gt['objects'][-1]['bbox'].tolist()
            obs_set_gt['objects'][-1]['pos'] = obs_set_gt['objects'][-1]['pos'].tolist()
            obs_set_gt['objects'][-1]['ori'] = obs_set_gt['objects'][-1]['ori'].tolist()
            if obs_set_gt['objects'][-1]['weight'] is not None:
                obs_set_gt['objects'][-1]['weight'] = obs_set_gt['objects'][-1]['weight'].tolist()


        info_struct['observations'].append(obs_set)
        info_struct['observations_gt'].append(obs_set_gt)
        with open(self.json_path, 'w') as f:
            json.dump(info_struct, f, indent=4)
        self.obs_num += 1

    def outcome_to_json(self, outcome):
        if not self.environment.blender_enabled:
            return
        with open(self.json_path, 'r') as f:
            info_struct = json.load(f)

        info_struct['result'] = outcome

        with open(self.json_path, 'w') as f:
            json.dump(info_struct, f, indent=4)

    def _detect_loop(self, target_list):
        if len(self.loop_detector) < 2:
            return False
        last_commands = self.loop_detector.get()
        for source_list in last_commands:
            if len(source_list) != len(target_list):
                continue
            else:
                same = True
                for i, command in enumerate(source_list):
                    if command != target_list[i]:
                        same = False
                        break
                if same:
                    return True
        return False

    def _check_gt_scene(self):
        for i in range(len(self.scene_graph_gt)):
            if self.scene_graph_gt[i]['pos'][2] < -0.1:
                return False
        return True