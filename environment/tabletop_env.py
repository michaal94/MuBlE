import random
from collections import OrderedDict

import os
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from .table_arena import TableArena
from robosuite.models.base import MujocoModel
from robosuite.models.grippers import GripperModel
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from environment.objects import CoordinateVis, ShopVrbObject
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.controllers import controller_factory, load_controller_config
from .controller import CustomOperationalSpaceController


class TabletopEnv(SingleArmEnv):
    """
    This class corresponds to the pick place task for a single robot arm.
    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!
        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.
        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param
        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param
        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:
            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"
            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param
            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.
        table_full_size (3-tuple): x, y, and z dimensions of the table.
        table_friction (3-tuple): the three mujoco friction parameters for
            the table.
        bin1_pos (3-tuple): Absolute cartesian coordinates of the bin initially holding the objects
        bin2_pos (3-tuple): Absolute cartesian coordinates of the goal bin
        use_camera_obs (bool): if True, every observation includes rendered image(s)
        use_object_obs (bool): if True, include object (cube) information in
            the observation.
        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized
        reward_shaping (bool): if True, use dense rewards.
        single_object_mode (int): specifies which version of the task to do. Note that
            the observations change accordingly.
            :`0`: corresponds to the full task with all types of objects.
            :`1`: corresponds to an easier task with only one type of object initialized
               on the table with every reset. The type is randomized on every reset.
            :`2`: corresponds to an easier task with only one type of object initialized
               on the table with every reset. The type is kept constant and will not
               change between resets.
        object_type (string): if provided, should be one of "milk", "bread", "cereal",
            or "can". Determines which type of object will be spawned on every
            environment reset. Only used if @single_object_mode is 2.
        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.
        has_offscreen_renderer (bool): True if using off-screen rendering
        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse
        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.
        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.
        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).
        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.
        horizon (int): Every episode lasts for exactly @horizon timesteps.
        ignore_done (bool): True if never terminating the environment (ignore @horizon).
        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables
        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.
            :Note: At least one camera must be specified if @use_camera_obs is True.
            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).
        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.
        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.
        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.
        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:
                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level
            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.
    Raises:
        AssertionError: [Invalid object type specified]
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        objs,
        robots,
        mode='train',
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(1.36, 0.8, 0.026),
        table_offset=(0, 0, 0.8),
        table_friction=(1, 0.005, 0.0001),
        use_camera_obs=True,
        use_object_obs=False,
        reward_scale=1.0,
        reward_shaping=False,
        object_type=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        bounding_boxes_from_scene=False,
        debug=False,
        blender_render=False,
        blender_config=None,
        scene_dict=None,
        output_render_path='./image',
        return_bboxes=False,
        list_obj_properties=False,
        use_custom_controller=True
    ):
        self.use_custom_controller = use_custom_controller
        if use_custom_controller:
            self.controller_cfg = controller_configs
        self.return_bboxes = return_bboxes
        self.list_obj_properties = list_obj_properties
        self.blender_enabled = blender_render
        if blender_render:
            assert blender_config is not None
            assert scene_dict is not None
            # Import here cause of bpy import which can go only when calling blender
            from .renderer import BlenderRenderer
            self.blender_renderer = BlenderRenderer(blender_config)
            self.blender_renderer.init_scene(scene_dict)
            self.blender_counter = 0

        self.gt_scene = scene_dict
        # task settings
        self.use_object_obs = use_object_obs
        self.bounding_boxes_from_scene = bounding_boxes_from_scene
        self.obj_names = list(objs.keys())
        self.object_to_id = dict(
            zip(
                self.obj_names,
                list(range(len(self.obj_names)))
            )
        )
        self.objs = objs
        self.mode = mode

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = table_offset

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        # self.use_object_obs = use_object_obs
        # if gripper_types == 'customRobotiq85Gripper':
        #     gripper_types = None
        #     self.internal_gripper_types = 'customRobotiq85Gripper'
        # else:
        #     self.internal_gripper_types = None

        # print(gripper_types)

        self.debug = debug

        self.grasped_obj = None
        self.grasped_obj_rel_mat = None
        self.release_delay = 1
        # self.release_delay = 1000
        self.release_delay_counter = None

        self.out_path = output_render_path

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types=None,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):

        return 0

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        self.robot_configs[0]['initial_qpos'] = np.array(
            [0.0, 0.0, 0.0, - 3 * np.pi / 4.0, 0.00, 3 * np.pi / 4, np.pi / 4]
        )



        # self.robot_configs[0]['initial_qpos'] = np.array(
        #     [0.000, 0.650, 0.000, 1.890, 0.000, 0.600, -np.pi / 2]
        # )

        # self.robot_configs[0]['initial_qpos'] = np.array(
        #     [0.0, 0.0, 0.0, 0, 0.00, 0, 0]
        # )
        # self.robot_configs[0]['initial_qpos'] = np.array(
        #     [np.pi / 4, 0.0, 0.0, - 3 * np.pi / 4.0, 0.00, 2 * np.pi / 4, 3 * np.pi / 4]
        # )
        # print(self.robot_configs)
        super()._load_model()

        xpos = (
            0,
            0,
            0)
        self.robots[0].robot_model.set_base_xpos(xpos)

        # print(self.robots[0].robot_model.eef_name)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction,
            table_offset=self.table_offset
        )

        mujoco_arena.set_origin([
            self.table_full_size[0] / 2 - 0.23 - self.table_offset[0],
            - self.table_offset[1],
            - self.table_offset[2]])

        self.objects = []

        for obj_name, obj_attrs in self.objs.items():
            obj = ShopVrbObject(
                obj_name,
                obj_attrs['file_name'],
                scale=obj_attrs['scale'],
                mode=self.mode
            ) 
            self.objects.append(obj)

        self.debug_objs = []
        if self.debug:
            # self.coord_systems = []
            for i in range(4):
                coord_sys = CoordinateVis(f'coord_sys_{i}')
                self.debug_objs.append(coord_sys)
        

        self.robots[0].gripper.current_action = [-1.0]

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects= self.objects + self.debug_objs,
        )

        # Generate placement initializer
        # self._get_placement_initializer()

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()
        # Additional object references from this env
        self.obj_body_id = {}
        self.obj_geom_id = {}

        self.obj_to_idx = {}

        # object-specific ids
        for i, obj in enumerate(self.objects):
            self.obj_to_idx[obj.name] = i
            self.obj_body_id[obj.name] = self.sim.model.body_name2id(obj.root_body)
            self.obj_geom_id[obj.name] = [self.sim.model.geom_name2id(g) for g in obj.contact_geoms]

        if self.debug:
            self.debug_obj_body_id = {}
            for do in self.debug_objs:
                self.debug_obj_body_id[do.name] = self.sim.model.body_name2id(do.root_body)


    def _setup_observables(self):
        observables = super()._setup_observables()

        # self.gripper_actuator_idxs = [self.sim.model.actuator_name2id(actuator) for actuator in self.robots[0].gripper.actuators]
        # self.gripper_ctrl_range = self.sim.model.actuator_ctrlrange[self.gripper_actuator_idxs]
        # print(self.gripper_ctrl_range)
        # exit()
        # gripper_actuators = [self.sim.data.actuator_length[i] for i in actuator_idxs]
        # print(gripper_actuators)
        @sensor(modality='gripper')
        def gripper_action(obs_cache):
            return self.robots[0].gripper.current_action[0]
        observables['gripper_action'] = Observable(
            name='gripper_action',
            sensor=gripper_action,
            sampling_rate=self.control_freq,
            enabled=1,
            active=1,
        )

        @sensor(modality='gripper')
        def gripper_closed(obs_cache):
            if self.grasped_obj is not None:
                return True
            else:
                return (self.robots[0].gripper.current_action[0] > 0.9)
        observables['gripper_closed'] = Observable(
            name='gripper_closed',
            sensor=gripper_closed,
            sampling_rate=self.control_freq,
            enabled=1,
            active=1,
        )

        @sensor(modality='weight')
        def weight_measurement(obs_cache):
            if self.grasped_obj is not None:
                return self.gt_scene['objects'][self.obj_to_idx[self.grasped_obj.name]]['weight_gt']
            else:
                return 0.0
        observables['weight_measurement'] = Observable(
            name='weight_measurement',
            sensor=weight_measurement,
            sampling_rate=self.control_freq,
            enabled=1,
            active=1,
        ) 

        if self.use_object_obs:
            # Get robot prefix and define observables modality
            # pf = self.robots[0].robot_model.naming_prefix
            
            if self.list_obj_properties:
                # Create all obj sensor
                obj_sensor, obj_sensor_name = self._create_list_obj_sensor(self.return_bboxes)
                observables[obj_sensor_name] = Observable(
                    name=obj_sensor_name,
                    sensor=obj_sensor,
                    sampling_rate=self.control_freq,
                    enabled=1,
                    active=1,
                )
            else:
                modality = "object"
                self.object_id_to_sensors = {}
                sensors = []
                names = []
                enableds = []
                actives = []

                for i, obj in enumerate(self.objects):
                    # Create object sensors
                    obj_sensors, obj_sensor_names = self._create_obj_sensors(obj=obj, modality=modality, add_bboxes=self.return_bboxes)
                    sensors += obj_sensors
                    names += obj_sensor_names
                    enableds += [1] * len(obj_sensor_names)
                    actives += [1] * len(obj_sensor_names)
                    self.object_id_to_sensors[i] = obj_sensor_names

                # Create observables
                for name, s, enabled, active in zip(names, sensors, enableds, actives):
                    observables[name] = Observable(
                        name=name,
                        sensor=s,
                        sampling_rate=self.control_freq,
                        enabled=enabled,
                        active=active,
                    )

            @sensor(modality='gripper')
            def grasped_obj(obs_cache):
                if self.grasped_obj is not None:
                    return self.grasped_obj.name
                else:
                    return None
            observables['grasped_obj'] = Observable(
                name='grasped_obj',
                sensor=grasped_obj,
                sampling_rate=self.control_freq,
                enabled=1,
                active=1,
            )

            @sensor(modality='gripper')
            def grasped_obj_idx(obs_cache):
                if self.grasped_obj is not None:
                    return self.obj_to_idx[self.grasped_obj.name]
                else:
                    return None
            observables['grasped_obj_idx'] = Observable(
                name='grasped_obj_idx',
                sensor=grasped_obj_idx,
                sampling_rate=self.control_freq,
                enabled=1,
                active=1,
            )

        return observables

    def _create_list_obj_sensor(self, add_bboxes=False):
        @sensor(modality='object_list')
        def obj_list_sensor(obs_cache):
            if add_bboxes:
                bboxes = self.get_bounding_boxes_xyz()
            obj_list = []
            for o in self.objects:
                obj_name = o.name
                pos = np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])
                ori = T.convert_quat(self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw")
                if add_bboxes:
                    bbox = bboxes[obj_name]
                    tup = (obj_name, pos, ori, bbox)
                else:
                    bbox = None
                    tup = (obj_name, pos, ori)
                # obj_dict = {
                #     "name": obj_name,
                #     "pos": pos,
                #     "quat": ori,
                #     "bbox": bbox
                # }
                obj_list.append(tup)
            return np.array(obj_list, dtype=object)

        return obj_list_sensor, 'objects'

    def _create_obj_sensors(self, obj, modality="object", add_bboxes=False):
    #     """
    #     Helper function to create sensors for a given object. This is abstracted in a separate function call so that we
    #     don't have local function naming collisions during the _setup_observables() call.
    #     Args:
    #         obj_name (str): Name of object to create sensors for
    #         modality (str): Modality to assign to all sensors
    #     Returns:
    #         2-tuple:
    #             sensors (list): Array of sensors for the given obj
    #             names (list): array of corresponding observable names
    #     """
        pf = self.robots[0].robot_model.naming_prefix

        obj_name = obj.name

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw")

        @sensor(modality='bbox')
        def obj_bbox(obs_cache):
            if add_bboxes:
                return self.get_bounding_box_xyz(obj)
            else:
                return None

        if add_bboxes:
            sensors = [obj_pos, obj_quat, obj_bbox]
            names = [f"{obj_name}_pos", f"{obj_name}_quat", f"{obj_name}_bbox"]
        else:
            sensors = [obj_pos, obj_quat]
            names = [f"{obj_name}_pos", f"{obj_name}_quat"]

        return sensors, names

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        for obj, attributes in zip(self.objects, self.objs.values()):
            self.sim.data.set_joint_qpos(
                obj.joints[0],
                np.concatenate(
                    [np.array(attributes['position']),
                    np.array(attributes['orientation'])]
                    )
            )

        if self.use_custom_controller:
            self._load_custom_controller()

        if self.debug:
            print('Debug')
            for i, c in enumerate(self.debug_objs):
                self.sim.model.body_pos[self.debug_obj_body_id[c.name]] = [0, 0, 0]
        
    def _pre_action(self, action, policy_step=False):
        """
        Overrides the superclass method to control the robot(s) within this enviornment using their respective
        controllers using the passed actions and gripper control.
        Args:
            action (np.array): The control to apply to the robot(s). Note that this should be a flat 1D array that
                encompasses all actions to be distributed to each robot if there are multiple. For each section of the
                action space assigned to a single robot, the first @self.robots[i].controller.control_dim dimensions
                should be the desired controller actions and if the robot has a gripper, the next
                @self.robots[i].gripper.dof dimensions should be actuation controls for the gripper.
            policy_step (bool): Whether a new policy step (action) is being taken
        Raises:
            AssertionError: [Invalid action dimension]
        """
        # Verify that the action is the correct dimension
        assert len(action) == self.action_dim, "environment got invalid action dimension -- expected {}, got {}".format(
            self.action_dim, len(action)
        )
        
        # print(np.array(self.sim.data.qvel[self.robots[0]._ref_joint_vel_indexes]))

        if self.debug:
            hand_pos = np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id])
            # hand_pos = np.array(self.sim.data.get_body_xpos(self.robots[0].robot_model.eef_name))
            hand_ori = self.sim.data.get_body_xquat(self.robots[0].robot_model.eef_name)
            # hand_pos = np.array(self.sim.data.site_xpos[self.sim.model.site_name2id('gripper0_test')])
            # hand_ori = T.mat2quat(np.array(self.sim.data.site_xmat[self.sim.model.site_name2id('gripper0_test')].reshape([3, 3])))
            # hand_ori = T.mat2quat(np.array(self.sim.data.get_body_xmat(self.robots[0].robot_model.eef_name)).reshape([3, 3]))
            self.sim.model.body_pos[self.debug_obj_body_id[self.debug_objs[0].name]] = hand_pos
            self.sim.model.body_quat[self.debug_obj_body_id[self.debug_objs[0].name]] = hand_ori

            finger_dist = 0.1113 / 2
            gripper_y_dir = T.quat2mat(hand_ori)[:, 1]
            gripper_y_dir = gripper_y_dir / np.linalg.norm(gripper_y_dir)
            finger1 = hand_pos + finger_dist * gripper_y_dir
            finger2 = hand_pos - finger_dist * gripper_y_dir

            self.sim.model.body_pos[self.debug_obj_body_id[self.debug_objs[2].name]] = finger1
            self.sim.model.body_quat[self.debug_obj_body_id[self.debug_objs[2].name]] = hand_ori
            self.sim.model.body_pos[self.debug_obj_body_id[self.debug_objs[3].name]] = finger2
            self.sim.model.body_quat[self.debug_obj_body_id[self.debug_objs[3].name]] = hand_ori

            # self.sim.model.body_pos[self.debug_obj_body_id[self.debug_objs[2].name]] = self.robots[0].controller.ee_pos + np.array([0, 0.2, 0])
            # self.sim.model.body_quat[self.debug_obj_body_id[self.debug_objs[2].name]] = T.convert_quat(T.mat2quat(self.robots[0].controller.ee_ori_mat), to='wxyz')
            # self.sim.model.body_pos[self.debug_obj_body_id[self.debug_objs[3].name]] = self.robots[0].controller.goal_pos + np.array([0, 0.2, 0])
            # self.sim.model.body_quat[self.debug_obj_body_id[self.debug_objs[3].name]] = T.convert_quat(T.mat2quat(self.robots[0].controller.goal_ori), to='wxyz')

            # self.sim.model.body_pos[self.debug_obj_body_id[self.debug_objs[2].name]] = self.robots[0].controller.ee_pos
            # self.sim.model.body_quat[self.debug_obj_body_id[self.debug_objs[2].name]] = self.robots[0].controller.ee_ori_quat
            # self.sim.model.body_pos[self.debug_obj_body_id[self.debug_objs[3].name]] = self.robots[0].controller.goal_pos
            # self.sim.model.body_quat[self.debug_obj_body_id[self.debug_objs[3].name]] = self.robots[0].controller.goal_ori_quat


        # Gripper state:
        # 1 - closing
        # -1 - opening
        gripper_state = 0 
        if action[6] > self.robots[0].gripper.current_action[0]:
            gripper_state = 1
        elif action[6] < self.robots[0].gripper.current_action[0] and action[6] < -0.99:
            gripper_state = -1

        # If gripper is closing
        if gripper_state == 1 or action[6] == 1:
            # Check if anything is grasped
            if self.grasped_obj is not None:
                # Keep current closure
                action[6] = self.robots[0].gripper.current_action[0]
            else:
                # Check if we 'start' grasping any object
                for i, o in enumerate(self.objects):
                    obj_grasped = self._check_grasp(self.robots[0].gripper, o)
                    # If we found touching obj, save it, save relative pos and ori w.r.t. gripper
                    if obj_grasped:
                        self.grasped_obj = o
                        action[6] = self.robots[0].gripper.current_action[0]
                        # print(o.name)
                        obj_world_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[o.name]])
                        obj_world_ori = T.convert_quat(self.sim.data.body_xquat[self.obj_body_id[o.name]], to="xyzw")
                        obj_world_mat = T.pose2mat((obj_world_pos, obj_world_ori))
                        eef_world_mat = self.robots[0]._hand_pose
                        world_eef_mat = T.pose_inv(eef_world_mat)
                        obj_eef_mat = T.pose_in_A_to_pose_in_B(obj_world_mat, world_eef_mat)
                        self.grasped_obj_rel_mat = obj_eef_mat            
        
        # If gripper is opening:
        if gripper_state == -1:
            # If anything is grasped release it
            if self.grasped_obj is not None:
                # Start delay
                self.release_delay_counter = self.release_delay

        if self.release_delay_counter is not None:
            self.release_delay_counter -= 1
            if self.release_delay_counter == 0:
                self.release_delay_counter = None
                self.grasped_obj = None
                self.grasped_obj_rel_pos = None
                self.grasped_obj_rel_ori = None


        # Update robot joints based on controller actions
        cutoff = 0
        for idx, robot in enumerate(self.robots):
            robot_action = action[cutoff : cutoff + robot.action_dim]
            robot.control(robot_action, policy_step=policy_step)
            cutoff += robot.action_dim

        if self.grasped_obj is not None:
            eef_world_mat = self.robots[0]._hand_pose
            obj_world_mat = T.pose_in_A_to_pose_in_B(self.grasped_obj_rel_mat, eef_world_mat)
            self.sim.data.set_joint_qpos(
                self.grasped_obj.joints[0],
                np.concatenate(
                    [
                        obj_world_mat[:3, 3],
                        T.convert_quat(T.mat2quat(obj_world_mat[:3, :3]), to="wxyz")
                    ]
                )
            )
            self.sim.data.set_joint_qvel(
                self.grasped_obj.joints[0],
                np.concatenate(
                    [
                        np.zeros((6))
                    ]
                )
            )


    def _check_grasp(self, gripper, object_geoms):
        """
        Checks whether the specified gripper as defined by @gripper is grasping the specified object in the environment.
        By default, this will return True if at least one geom in both the "left_fingerpad" and "right_fingerpad" geom
        groups are in contact with any geom specified by @object_geoms. Custom gripper geom groups can be
        specified with @gripper as well.
        Args:
            gripper (GripperModel or str or list of str or list of list of str): If a MujocoModel, this is specific
            gripper to check for grasping (as defined by "left_fingerpad" and "right_fingerpad" geom groups). Otherwise,
                this sets custom gripper geom groups which together define a grasp. This can be a string
                (one group of single gripper geom), a list of string (multiple groups of single gripper geoms) or a
                list of list of string (multiple groups of multiple gripper geoms). At least one geom from each group
                must be in contact with any geom in @object_geoms for this method to return True.
            object_geoms (str or list of str or MujocoModel): If a MujocoModel is inputted, will check for any
                collisions with the model's contact_geoms. Otherwise, this should be specific geom name(s) composing
                the object to check for contact.
        Returns:
            bool: True if the gripper is grasping the given object
        """
        # Convert object, gripper geoms into standardized form
        if isinstance(object_geoms, MujocoModel):
            o_geoms = object_geoms.contact_geoms
        else:
            o_geoms = [object_geoms] if type(object_geoms) is str else object_geoms
        if isinstance(gripper, GripperModel):
            g_geoms = [gripper.important_geoms["left_fingerpad"], gripper.important_geoms["right_fingerpad"]]
        elif type(gripper) is str:
            g_geoms = [[gripper]]
        else:
            # Parse each element in the gripper_geoms list accordingly
            g_geoms = [[g_group] if type(g_group) is str else g_group for g_group in gripper]

        # Search for collisions between each gripper geom group and the object geoms group
        for g_group in g_geoms:
            clist = []
            for contact in self.sim.data.contact[: self.sim.data.ncon]:
                g1, g2 = self.sim.model.geom_id2name(contact.geom1), self.sim.model.geom_id2name(contact.geom2)
                clist.append((g1, g2))
            if not self.check_contact(g_group, o_geoms):
                return False
        return True

    def _check_success(self):
        return False

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the closest object.
        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

    def get_bounding_boxes_xyz(self):
        bboxes = {}
        for o in self.objects:
            bboxes[o.name] = self.get_bounding_box_xyz(o)
        return bboxes

    def get_bounding_box_xyz(self, obj):
        if self.bounding_boxes_from_scene:
            bbox_local = self.objs[obj.name]['bbox']
            bbox_local = np.concatenate(
                (
                    bbox_local.T,
                    np.ones((bbox_local.shape[0], 1)).T
                )
            )
            obj_world_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[obj.name]])
            obj_world_ori = T.convert_quat(self.sim.data.body_xquat[self.obj_body_id[obj.name]], to="xyzw")
            obj_world_mat = T.pose2mat((obj_world_pos, obj_world_ori))
            bbox_world = np.matmul(obj_world_mat, bbox_local)
            return bbox_world[:-1, :].T
        else:
            obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj.name]]
            contact_geoms = self.obj_geom_id[obj.name]
            geom_pos = [self.sim.data.geom_xpos[i] for i in contact_geoms]    
            geom_rbound = [self.sim.model.geom_rbound[i] for i in contact_geoms]
            geom_max_bounds = [geom_pos[i] + geom_rbound[i] for i in range(len(geom_pos))]
            geom_min_bounds = [geom_pos[i] - geom_rbound[i] for i in range(len(geom_pos))]
            z_bound_top = max([g[2] for g in geom_max_bounds]) - obj_pos[2]
            z_bound_bot = max(min([g[2] for g in geom_min_bounds]), 0) - obj_pos[2]
            y_bound_top = max([g[1] for g in geom_max_bounds]) - obj_pos[1]
            y_bound_bot = min([g[1] for g in geom_min_bounds]) - obj_pos[1]
            x_bound_top = max([g[0] for g in geom_max_bounds]) - obj_pos[0]
            x_bound_bot = min([g[0] for g in geom_min_bounds]) - obj_pos[0]
            return np.array(
                [
                    [x_bound_top, y_bound_top, z_bound_bot],
                    [x_bound_bot, y_bound_top, z_bound_bot],
                    [x_bound_bot, y_bound_bot, z_bound_bot],
                    [x_bound_top, y_bound_bot, z_bound_bot],
                    [x_bound_top, y_bound_top, z_bound_top],
                    [x_bound_bot, y_bound_top, z_bound_top],
                    [x_bound_bot, y_bound_bot, z_bound_top],
                    [x_bound_top, y_bound_bot, z_bound_top]
                ]
            ) + obj_pos

    def print_robot_configuration(self):
        robot, gripper = self.get_robot_configuration()
        print(robot)
        print(gripper)
    
    def get_robot_configuration(self):
        robot = {}
        for i in range(8):
            robot[f'link{i}'] = (
                self.sim.data.get_body_xpos(f'robot0_link{i}'),
                self.sim.data.get_body_xquat(f'robot0_link{i}')
            )

        gripper = {}
        gripper['robotiq_85_adapter_link'] = (
            self.sim.data.get_body_xpos('gripper0_robotiq_85_adapter_link'),
            self.sim.data.get_body_xquat('gripper0_robotiq_85_adapter_link')
        )
        gripper['left_inner_finger'] = (
            self.sim.data.get_body_xpos('gripper0_left_inner_finger'),
            self.sim.data.get_body_xquat('gripper0_left_inner_finger')
        )
        gripper['left_inner_knuckle'] = (
            self.sim.data.get_body_xpos('gripper0_left_inner_knuckle'),
            self.sim.data.get_body_xquat('gripper0_left_inner_knuckle')
        )
        gripper['left_outer_knuckle'] = (
            self.sim.data.get_body_xpos('gripper0_left_outer_knuckle'),
            self.sim.data.get_body_xquat('gripper0_left_outer_knuckle')
        )
        gripper['right_inner_finger'] = (
            self.sim.data.get_body_xpos('gripper0_right_inner_finger'),
            self.sim.data.get_body_xquat('gripper0_right_inner_finger')
        )
        gripper['right_inner_knuckle'] = (
            self.sim.data.get_body_xpos('gripper0_right_inner_knuckle'),
            self.sim.data.get_body_xquat('gripper0_right_inner_knuckle')
        )
        gripper['right_outer_knuckle'] = (
            self.sim.data.get_body_xpos('gripper0_right_outer_knuckle'),
            self.sim.data.get_body_xquat('gripper0_right_outer_knuckle')
        )
        return robot, gripper

    def get_segmentation_masks(self):
        if self.blender_enabled:
            return self.blender_renderer.get_segmentation_masks()
        else:
            return None

    def blender_render(self):
        if self.blender_enabled:
            objects = {}
            for o in self.objects:
                pos = self.sim.data.body_xpos[self.obj_body_id[o.name]]
                ori = self.sim.data.body_xquat[self.obj_body_id[o.name]]
                objects[o.name] = (pos, ori)

            robot = {}
            for i in range(8):
                robot[f'link{i}'] = (
                    self.sim.data.get_body_xpos(f'robot0_link{i}'),
                    self.sim.data.get_body_xquat(f'robot0_link{i}')
                )

            gripper = {}
            gripper['robotiq_85_adapter_link'] = (
                self.sim.data.get_body_xpos('gripper0_robotiq_85_adapter_link'),
                self.sim.data.get_body_xquat('gripper0_robotiq_85_adapter_link')
            )
            gripper['left_inner_finger'] = (
                self.sim.data.get_body_xpos('gripper0_left_inner_finger'),
                self.sim.data.get_body_xquat('gripper0_left_inner_finger')
            )
            gripper['left_inner_knuckle'] = (
                self.sim.data.get_body_xpos('gripper0_left_inner_knuckle'),
                self.sim.data.get_body_xquat('gripper0_left_inner_knuckle')
            )
            gripper['left_outer_knuckle'] = (
                self.sim.data.get_body_xpos('gripper0_left_outer_knuckle'),
                self.sim.data.get_body_xquat('gripper0_left_outer_knuckle')
            )
            gripper['right_inner_finger'] = (
                self.sim.data.get_body_xpos('gripper0_right_inner_finger'),
                self.sim.data.get_body_xquat('gripper0_right_inner_finger')
            )
            gripper['right_inner_knuckle'] = (
                self.sim.data.get_body_xpos('gripper0_right_inner_knuckle'),
                self.sim.data.get_body_xquat('gripper0_right_inner_knuckle')
            )
            gripper['right_outer_knuckle'] = (
                self.sim.data.get_body_xpos('gripper0_right_outer_knuckle'),
                self.sim.data.get_body_xquat('gripper0_right_outer_knuckle')
            )

            self.blender_renderer.update_scene(
                objects,
                robot,
                gripper
            )
            path = f'{self.out_path}_{self.blender_counter:04d}.png'
            self.blender_renderer.render(path)
            self.blender_counter += 1
            return path

    def get_default_action_abs(self):
        hand_pos = np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id])
        hand_ori = T.convert_quat(self.sim.data.get_body_xquat(self.robots[0].robot_model.eef_name), to='xyzw')
        hand_ori = T.quat2axisangle(hand_ori)
        return [hand_pos[0], hand_pos[1], hand_pos[2], hand_ori[0], hand_ori[1], hand_ori[2], -1]

    def _load_custom_controller(self):
        cfg = self.controller_cfg
        self.controller_cfg["robot_name"] = self.robots[0].name
        self.controller_cfg["sim"] = self.sim
        self.controller_cfg["eef_name"] = self.robots[0].robot_model.eef_name
        self.controller_cfg["eef_rot_offset"] = self.robots[0].eef_rot_offset
        self.controller_cfg["joint_indexes"] = {
            "joints": self.robots[0].joint_indexes,
            "qpos": self.robots[0]._ref_joint_pos_indexes,
            "qvel": self.robots[0]._ref_joint_vel_indexes,
        }
        self.controller_cfg["actuator_range"] = self.robots[0].torque_limits
        self.controller_cfg["policy_freq"] = self.robots[0].control_freq
        self.controller_cfg["ndim"] = len(self.robots[0].robot_joints)

        if 'custom_type' in self.controller_cfg:
            name = self.controller_cfg['custom_type']
        else:
            name = self.controller_cfg['type']
        if name == 'OSC_POSE':
            controller = CustomOperationalSpaceController(**self.controller_cfg)
        elif name == 'NEO':
            controller = NeoController(**self.controller_cfg)

        self.robots[0].controller = controller