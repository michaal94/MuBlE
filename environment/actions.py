from copy import copy
import numpy as np
import robosuite.utils.transform_utils as T
from utils.utils import CyclicBuffer

class ActionExecutor:
    def __init__(
        self,
        env=None,
        delta_control=True,
        gt_bounding_boxes=None,
        move_clearance=0.1,
        pick_up_height=0.1
    ) -> None:
        self.action = None
        self.move_dict = None
        self.eps_grasp = 0.001
        self.eps_pos = 0.002
        self.eps_move_end = 0.0075
        self.eps_move_trajectory = 0.008
        self.move_clearance = move_clearance
        self.pick_up_height = pick_up_height
        self.angle_up_tolerance = np.pi / 6
        self.angle_diff_tol = np.pi / 45
        self.eps_move_l1_ori = np.pi / 180
        self.grasp_depth = 0.02
        self.env = env
        self.grasp_buffer = None
        self.grasp_buffer_size = 5
        self.grasp_target = None
        self.gripper_dir = -1.0
        self.object_in_grasp = None
        self.obj_put_down_clearence = 0.02
        # Half of the pad:
        self.gripper_pad_height = 0.016
        self.gripper_open_width = 0.1113 / 2
        self.ori_sensitivity = 0.01
        self.delta_control = delta_control
        self.interp_angle = np.pi / 15
        self.interp_dist = 0.05
        self.max_interp = 0

    def set_action(self, action, target, obs, scene):
        self.action = action
        if action == 'move':
            self._set_move(target, obs, scene)
        elif self.action == 'approach_grasp':
            self._set_approach_grasp(target, obs, scene)
        elif self.action == 'grasp':
            self._set_grasp(target, obs, scene)
        elif self.action == 'release':
            self._set_release(target, obs, scene)
        elif self.action == 'pick_up':
            self._set_pick_up(target, obs, scene)
        elif self.action == 'put_down':
            self._set_put_down(target, obs, scene)

    def _unset_action(self):
        self.action = None
        self.move_dict = None
        self.grasp_buffer = None
        self.grasp_target = None

    def step(self, obs):
        if self.action is None:
            return [0, 0, 0, 0, 0, 0, self.gripper_dir]
        else:
            if self.action == 'move':
                action = self._move(obs)
            elif self.action == 'approach_grasp':
                action = self._move(obs)
            elif self.action == 'grasp':
                action = self._grasp(obs)
            elif self.action == 'release':
                action = self._release(obs)
            elif self.action == 'pick_up':
                action = self._move(obs)
            elif self.action == 'put_down':
                action = self._move(obs)

        return action

    def _set_grasp(self, target, obs, scene):
        self.grasp_buffer = CyclicBuffer(self.grasp_buffer_size)
        self.grasp_target = target
        self.object_in_grasp = target
        self.gripper_dir = 1.0

    def _set_release(self, target, obs, bboxes):
        self.gripper_dir = -1.0
        self.object_in_grasp = None

    def _set_move(self, target, obs, scene):
        self.move_dict = {}
        if target in ['left', 'right', 'front', 'back']:
            self._set_move_table_part(target, obs, scene)
        else:
            obj_clearance = 0.0
            in_hand = None
            eef_to_in_hand = np.array([0.0, 0.0, 0.0])
            for idx, obj in enumerate(scene):
                if obj['in_hand']:
                    in_hand = idx
            if in_hand is not None:
                move_clearance = 0.075
                # Moving with sth in hand
                bbox_in_hand = scene[in_hand]['bbox']
                obj_clearance = np.abs(
                    # obs['robot0_eef_pos'] - min([bbox_in_hand[i][2] for i in range(8)])
                    obs['robot0_eef_pos'][2] - np.amin(bbox_in_hand[:, 2])
                )
                eef_to_in_hand = scene[in_hand]['pos'] - obs['robot0_eef_pos']
            else:
                move_clearance = self.move_clearance

            bbox = scene[target]['bbox']
            z_coord = bbox[:, 2]
            z_coord = np.amax(z_coord) + move_clearance + obj_clearance
            target_pos = scene[target]['pos'] - eef_to_in_hand
            target_pos[2] = z_coord
            self.move_dict['trajectory'] = self._get_move_trajectory(target_pos, obs, scene)
            self.move_dict['current_target'] = self.move_dict['trajectory'].pop(0)

    def _set_pick_up(self, target, obs, scene):
        self.move_dict = {}
        eef_pos = obs['robot0_eef_pos']
        eef_ori = obs['robot0_eef_quat']
        target_pos = eef_pos + np.array([0, 0, self.pick_up_height])
        self.move_dict['trajectory'] = [(target_pos, eef_ori)]
        self.move_dict['current_target'] = self.move_dict['trajectory'].pop(0)

    def _set_put_down(self, target, obs, scene):
        self.move_dict = {}
        bbox = scene[target]['bbox']
        z_coords = bbox[:, 2]
        min_z = np.amin(z_coords)
        obj_below_clearance = self._get_below_clearance(target, scene)
        if obj_below_clearance > 0:
            put_down_clearance = 0.0
        else:
            put_down_clearance = self.obj_put_down_clearence
        downward_movement = min_z - put_down_clearance - obj_below_clearance
        # print(min_z)
        # print(scene[target]['pos'])
        # print(obj_below_clearance)
        eef_pos = obs['robot0_eef_pos']
        eef_ori = obs['robot0_eef_quat']
        target_pos = eef_pos - np.array([0, 0, downward_movement])
        intermediate_pos = target_pos + np.array([0, 0, 0.01])
        self.move_dict['trajectory'] = [
            (intermediate_pos, eef_ori),
            (target_pos, eef_ori)
        ]
        self.move_dict['current_target'] = self.move_dict['trajectory'].pop(0)
    
    def _get_below_clearance(self, target, scene):
        clearance = 0.0
        target_pos = scene[target]['pos']
        target_pos_xy = Point(
            target_pos[0],
            target_pos[1]
        )
        for idx, obj in enumerate(scene):
            # print(obj['file'])
            # print(target)
            if idx == target:
                continue
            obj_z_dir = T.quat2mat(obj['ori'])[:, 2]
            # PRINT HERE
            # print(obj_z_dir)
            if self._angle_between(obj_z_dir, np.array([0, 0, 1])) < self.angle_up_tolerance:
                rec = Rectangle(
                    Point(obj['bbox'][0][0], obj['bbox'][0][1]),
                    Point(obj['bbox'][1][0], obj['bbox'][1][1]),
                    Point(obj['bbox'][2][0], obj['bbox'][2][1]),
                    Point(obj['bbox'][3][0], obj['bbox'][3][1])
                )
            else:
                rec1 = Rectangle(
                    Point(obj['bbox'][0][0], obj['bbox'][0][1]),
                    Point(obj['bbox'][1][0], obj['bbox'][1][1]),
                    Point(obj['bbox'][5][0], obj['bbox'][5][1]),
                    Point(obj['bbox'][4][0], obj['bbox'][4][1])
                )
                rec2 = Rectangle(
                    Point(obj['bbox'][0][0], obj['bbox'][0][1]),
                    Point(obj['bbox'][4][0], obj['bbox'][4][1]),
                    Point(obj['bbox'][7][0], obj['bbox'][7][1]),
                    Point(obj['bbox'][3][0], obj['bbox'][3][1])
                )
                if rec1.area() > rec2.area():
                    rec = rec1
                else:
                    rec = rec2
            height = np.amax(obj['bbox'][:, 2])
            if rec.contains(target_pos_xy):
                if height > clearance:
                    clearance = height
        return clearance

    def _set_move_table_part(self, target, obs, scene):
        self.move_dict = {}
        in_hand = None
        if target == "left":
            x_range = [0.3, 0.7]
            y_range = [-0.33, 0]
        elif target == "right":
            x_range = [0.3, 0.7]
            y_range = [0.0, 0.33]
        else:
            raise NotImplementedError()

        zone_rectangle = Rectangle(
            Point(x_range[0], y_range[0]),
            Point(x_range[1], y_range[0]),
            Point(x_range[1], y_range[1]),
            Point(x_range[0], y_range[1])
        )
        mid_zone = zone_rectangle.mid_point()

        for idx, obj in enumerate(scene):
            if obj['in_hand']:
                in_hand = idx
        if in_hand is None:
            target_pos = np.array(
                [mid_zone.x, mid_zone.y, self.move_clearance + 0.05]
            )
            self.move_dict['trajectory'] = self._get_move_trajectory(target_pos, obs, scene)
            self.move_dict['current_target'] = self.move_dict['trajectory'].pop(0)
        else:
            obj_stationary = []
            for i in range(len(scene)):
                if i != in_hand:
                    obj_stationary.append(i)
            stat_rectangles = []
            for idx in obj_stationary:
                bbox = np.array(scene[idx]['bbox'])
                rec = Rectangle(
                    Point(bbox[0][0], bbox[0][1]),
                    Point(bbox[1][0], bbox[1][1]),
                    Point(bbox[2][0], bbox[2][1]),
                    Point(bbox[3][0], bbox[3][1])
                )
                stat_rectangles.append(rec)
            target_org_bbox = np.array(scene[in_hand]['bbox'])
            target_org_rec = Rectangle(
                Point(target_org_bbox[0][0], target_org_bbox[0][1]),
                Point(target_org_bbox[1][0], target_org_bbox[1][1]),
                Point(target_org_bbox[2][0], target_org_bbox[2][1]),
                Point(target_org_bbox[3][0], target_org_bbox[3][1])
            )
            mid_target = target_org_rec.mid_point()
            mid_target_np = np.array((mid_target.x, mid_target.y))
            new_points = []
            for side in target_org_rec.sides:
                p = side.ps
                mid_p_vec = np.array((p.x, p.y)) - mid_target_np
                mid_new_vec = mid_p_vec + 0.015 * mid_p_vec / np.linalg.norm(mid_p_vec)
                new_p = mid_target_np + mid_new_vec
                new_points.append(Point(new_p[0], new_p[1]))
            target_bigger_rec = Rectangle(
                new_points[0],
                new_points[1],
                new_points[2],
                new_points[3]
            )
            mid_zone_np = np.array((mid_zone.x, mid_zone.y))
            r_max = np.linalg.norm(np.array((x_range[0], y_range[0])) - mid_zone_np)
            k = 0.01
            dist = 0
            theta = 0
            theta_inc = np.pi / 45
            list_len = 10
            prev_pos = np.array([100.0, 100.0])

            def check_position(src_rectangle, obstacle_rectangles):
                mid = src_rectangle.mid_point()
                for rec in obstacle_rectangles:
                    mid_rec = rec.mid_point()
                    if rec.contains(mid):
                        return False, None
                    if src_rectangle.contains(mid_rec):
                        return False, None
                for side in src_rectangle.sides:
                    for rec in obstacle_rectangles:
                        if rec.intersects(side):
                            return False, None
                closest_distance = None
                for rec in obstacle_rectangles:
                    mid_rec_min = rec.dist_to_point(mid)
                    if closest_distance is None:
                        closest_distance = mid_rec_min
                    else:
                        if mid_rec_min < closest_distance:
                            closest_distance = mid_rec_min
                return True, closest_distance

            ori_checks = [-np.pi / 4, -np.pi / 8, 0, np.pi / 8, np.pi / 4]
            found_list = []

            while dist < r_max:
                x0 = k * theta * np.cos(theta)
                y0 = k * theta * np.sin(theta)
                x = x0 + mid_zone.x
                y = y0 + mid_zone.y
                
                dist = np.linalg.norm(np.array((x, y)) - mid_zone_np)

                if np.linalg.norm(prev_pos - np.array([x, y])) < 0.015:
                    theta += theta_inc
                    continue
                prev_pos = np.array([x, y])

                if zone_rectangle.contains(Point(x, y)):
                    for angle in ori_checks:
                        new_points_rectangle = []
                        for side in target_bigger_rec.sides:
                            p = side.ps
                            px_org = p.x - mid_target.x
                            py_org = p.y - mid_target.y
                            x_test = px_org * np.cos(angle) - py_org * np.sin(angle) + x
                            y_test = px_org * np.sin(angle) + py_org * np.cos(angle) + y
                            new_points_rectangle.append(
                                Point(x_test, y_test)
                            )
                        test_rectangle = Rectangle(
                            new_points_rectangle[0],
                            new_points_rectangle[1],
                            new_points_rectangle[2],
                            new_points_rectangle[3]
                        )
                        obst_avoided, closest_dist = check_position(test_rectangle, stat_rectangles)
                        if obst_avoided:
                            pos_found = Point(x, y)
                            found_list.append((pos_found, closest_dist, angle))
                
                if len(found_list) > list_len:
                    break

                x = -x0 + mid_zone.x
                y = -y0 + mid_zone.y

                if zone_rectangle.contains(Point(x, y)):
                    for angle in ori_checks:
                        new_points_rectangle = []
                        for side in target_bigger_rec.sides:
                            p = side.ps
                            px_org = p.x - mid_target.x
                            py_org = p.y - mid_target.y
                            x_test = px_org * np.cos(angle) - py_org * np.sin(angle) + x
                            y_test = px_org * np.sin(angle) + py_org * np.cos(angle) + y
                            new_points_rectangle.append(
                                Point(x_test, y_test)
                            )
                        test_rectangle = Rectangle(
                            new_points_rectangle[0],
                            new_points_rectangle[1],
                            new_points_rectangle[2],
                            new_points_rectangle[3]
                        )

                        obst_avoided, closest_dist = check_position(test_rectangle, stat_rectangles)

                        if obst_avoided:
                            pos_found = Point(x, y)
                            found_list.append((pos_found, closest_dist, angle))
                
                if len(found_list) > list_len:
                    break

                theta += theta_inc
            
            if len(found_list) > 0:
                self.move_dict['trajectory'] = []
                found_list_sorted = sorted(found_list, key=lambda element: element[1])
                best = found_list_sorted[-1]
                best_point = best[0]
                angle = best[2]
                eef_pos = obs['robot0_eef_pos']
                eef_ori = obs['robot0_eef_quat']
                target_pos = np.array(
                    [best_point.x, best_point.y, eef_pos[2]]
                )
                in_between_quat = []
                if angle != 0:
                    rot_quat = np.array(
                        [
                            0,
                            0, 
                            np.sin(angle / 2),
                            np.cos(angle / 2)
                        ]
                    )
                    target_quat = T.quat_multiply(eef_ori, rot_quat)
                    in_between_quat = self._get_quat_interp(
                        eef_ori,
                        target_quat,
                        angle = self.interp_angle
                    )
                else:
                    target_quat = eef_ori
                for q in in_between_quat:
                    self.move_dict['trajectory'].append((
                        eef_pos, q.copy()
                    ))
                self.move_dict['trajectory'].append(
                    (eef_pos, target_quat.copy())
                )
                move_traj = self._get_move_trajectory(target_pos, obs, scene)
                for elem in move_traj:
                    self.move_dict['trajectory'].append(
                        (elem[0], target_quat.copy())
                    )
                self.move_dict['current_target'] = self.move_dict['trajectory'].pop(0)

            else:
                eef_pos = obs['robot0_eef_pos']
                eef_ori = obs['robot0_eef_quat']
                self.move_dict['trajectory'] = [(eef_pos, eef_ori)]
                self.move_dict['current_target'] = self.move_dict['trajectory'].pop(0)


    def _move(self, obs):
        eef_pos = obs['robot0_eef_pos']
        eef_ori = obs['robot0_eef_quat']
        curr_target_pos = self.move_dict['current_target'][0]
        curr_target_ori = self.move_dict['current_target'][1]
        curr_target_ori = self._unit_vector(curr_target_ori)

        # Debug
        self.env.sim.model.body_pos[self.env.debug_obj_body_id[self.env.debug_objs[1].name]] = curr_target_pos
        self.env.sim.model.body_quat[self.env.debug_obj_body_id[self.env.debug_objs[1].name]] = T.convert_quat(curr_target_ori, to='wxyz')
        
        pos_diff = curr_target_pos - eef_pos
        diff_pos = np.linalg.norm(pos_diff)
        if len(self.move_dict['trajectory']):
            eps_pos = self.eps_move_trajectory
        else:
            eps_pos = self.eps_move_end

        ori_diff = T.get_orientation_error(curr_target_ori, eef_ori)

        quat_diff = T.quat_inverse(eef_ori)
        quat_diff = T.quat_multiply(quat_diff, curr_target_ori)
        ori_diff = T.quat2axisangle(quat_diff)
        # print("ori_diff")
        # print('ori_diff', ori_diff, self._get_euler_diff_between_quat(eef_ori, curr_target_ori))
        # ori_diff = self._get_euler_diff_between_quat(eef_ori, curr_target_ori)
        # print("ori_diff", ori_diff)

        
        # curr_euler = T.quat2axisangle(eef_ori)
        # goal_ori = curr_euler + ori_diff
        quat_error = T.axisangle2quat(ori_diff)

        rotation_mat_error = T.quat2mat(quat_error)
        goal_orientation = np.dot(rotation_mat_error, T.quat2mat(eef_ori))
        goal_quat = T.mat2quat(goal_orientation)
        goal_quat = T.quat_multiply(eef_ori, T.axisangle2quat(ori_diff))

        self.env.sim.model.body_quat[self.env.debug_obj_body_id[self.env.debug_objs[1].name]] = T.convert_quat(goal_quat, to='wxyz')
        
        # if ori_diff[2] < -0.05:
        #     ori_diff[2] = - ori_diff[2]

        # if any([d > 2 * self.interp_angle for d in ori_diff]):
        #     interp_q = self._get_quat_interp(eef_ori, curr_target_ori, self.interp_angle)
        #     for q in reversed(interp_q[1:]):
        #         self.move_dict['trajectory'].insert(0, 
        #             (curr_target_pos, q)
        #         )
        #     curr_target_ori = interp_q[0]
        #     self.move_dict['current_target'] = (curr_target_pos, curr_target_ori)
        #     ori_diff = T.get_orientation_error(curr_target_ori, eef_ori)

        ori_diff_calc = self._get_euler_diff_between_quat(eef_ori, curr_target_ori)
        diff_ori = np.sum(np.abs(ori_diff_calc))
        # print('asdf', curr_target_pos, curr_target_ori)
        # print('ori_diff', ori_diff)
        # print(pos_diff)
        # print(diff_pos)
        # print(diff_ori)
        eps_ori = self.eps_move_l1_ori
        # print(eps_ori)

        # if len(self.move_dict['trajectory']) > self.max_interp:
        #     self.max_interp = len(self.move_dict['trajectory'])
        #     print(self.max_interp)
        # print(len(self.move_dict['trajectory']))

        if self.delta_control:
            if diff_pos < eps_pos and diff_ori < eps_ori:
                # print('both good')
                if len(self.move_dict['trajectory']) > 0:
                    new_targ = self.move_dict['trajectory'].pop(0)
                    self.move_dict['current_target'] = new_targ
                    pos_diff = new_targ[0] - eef_pos
                    ori_diff = T.get_orientation_error(new_targ[1], eef_ori)
                    return [0, 0, 0, ori_diff[0], ori_diff[1], ori_diff[2], self.gripper_dir]
                    # return [pos_diff[0], pos_diff[1], pos_diff[2], ori_diff[0], ori_diff[1], ori_diff[2], self.gripper_dir]
                else:
                    # print('done')
                    self._unset_action()
                    return [0, 0, 0, 0, 0, 0, self.gripper_dir]
            else:
                # return [pos_diff[0], pos_diff[1], pos_diff[2], ori_diff[0], ori_diff[1], ori_diff[2], self.gripper_dir]
                if diff_ori < eps_ori:
                    # print('pos')
                    return [pos_diff[0], pos_diff[1], pos_diff[2], 0, 0, 0, self.gripper_dir]
                else:
                    # print('ori')
                    # return [pos_diff[0], pos_diff[1], pos_diff[2], ori_diff[0], ori_diff[1], ori_diff[2], self.gripper_dir]
                    return [0, 0, 0, ori_diff[0], ori_diff[1], ori_diff[2], self.gripper_dir]
        else:
            curr_ori = T.quat2axisangle(eef_ori)
            # return [eef_pos[0], eef_pos[1], eef_pos[2], curr_ori[0], curr_ori[1], curr_ori[2], self.gripper_dir]
            if diff_pos < eps_pos and diff_ori < eps_ori:
                if len(self.move_dict['trajectory']) > 0:
                    new_targ = self.move_dict['trajectory'].pop(0)
                    self.move_dict['current_target'] = new_targ
                    new_pos = new_targ[0]
                    new_ori = T.quat2axisangle(new_targ[1])
                    # new_ori = T.quat2axisangle(eef_ori)
                    return [new_pos[0], new_pos[1], new_pos[2], new_ori[0], new_ori[1], new_ori[2], self.gripper_dir]
                else:
                    self._unset_action()
                    curr_ori = T.quat2axisangle(eef_ori)
                    return [eef_pos[0], eef_pos[1], eef_pos[2], curr_ori[0], curr_ori[1], curr_ori[2], self.gripper_dir]
            else:
                curr_ori = T.quat2axisangle(eef_ori)
                if diff_ori < eps_ori:
                    return [curr_target_pos[0], curr_target_pos[1], curr_target_pos[2], curr_ori[0], curr_ori[1], curr_ori[2], self.gripper_dir]
                else:
                    target_ori = T.quat2axisangle(curr_target_ori)
                    # target_ori = T.quat2axisangle(eef_ori)
                    return [curr_target_pos[0], curr_target_pos[1], curr_target_pos[2], target_ori[0], target_ori[1], target_ori[2], self.gripper_dir]
        
    def _grasp(self, obs):
        # gripper_closure = obs['gripper_action']
        # self.grasp_buffer.append(gripper_closure)
        # gripper_values = self.grasp_buffer.get()
        # # print(gripper_values)
        # if len(gripper_values) == self.grasp_buffer_size:
        #     gripper_values = np.array(gripper_values)
        #     diffs = gripper_values[1:] - gripper_values[:-1]
        #     diffs = np.abs(diffs)
        #     # print(diffs)
        #     max_diff = np.max(diffs)
        #     if max_diff < self.eps_grasp:
        #         self._unset_action()
        #         if self.delta_control:
        #             return [0, 0, 0, 0, 0, 0, self.gripper_dir]
        #         else:
        #             eef_pos = obs['robot0_eef_pos']
        #             eef_ori = obs['robot0_eef_quat']
        #             curr_ori = T.quat2axisangle(eef_ori)
        #             return [eef_pos[0], eef_pos[1], eef_pos[2], curr_ori[0], curr_ori[1], curr_ori[2], self.gripper_dir]
        gripper_closed = obs['gripper_closed']
        if gripper_closed:
            self._unset_action()
            if self.delta_control:
                return [0, 0, 0, 0, 0, 0, self.gripper_dir]
            else:
                eef_pos = obs['robot0_eef_pos']
                eef_ori = obs['robot0_eef_quat']
                curr_ori = T.quat2axisangle(eef_ori)
                return [eef_pos[0], eef_pos[1], eef_pos[2], curr_ori[0], curr_ori[1], curr_ori[2], self.gripper_dir]
        if self.delta_control:
            return [0, 0, 0, 0, 0, 0, 1]
        else:
            eef_pos = obs['robot0_eef_pos']
            eef_ori = obs['robot0_eef_quat']
            curr_ori = T.quat2axisangle(eef_ori)
            return [eef_pos[0], eef_pos[1], eef_pos[2], curr_ori[0], curr_ori[1], curr_ori[2], 1]

    def _release(self, obs):
        if obs['gripper_action'] < -0.99:
            self._unset_action()
            if self.delta_control:
                return [0, 0, 0, 0, 0, 0, self.gripper_dir]
            else:
                eef_pos = obs['robot0_eef_pos']
                eef_ori = obs['robot0_eef_quat']
                curr_ori = T.quat2axisangle(eef_ori)
                return [eef_pos[0], eef_pos[1], eef_pos[2], curr_ori[0], curr_ori[1], curr_ori[2], self.gripper_dir]
        if self.delta_control:
            return [0, 0, 0, 0, 0, 0, -1]
        else:
            eef_pos = obs['robot0_eef_pos']
            eef_ori = obs['robot0_eef_quat']
            curr_ori = T.quat2axisangle(eef_ori)
            return [eef_pos[0], eef_pos[1], eef_pos[2], curr_ori[0], curr_ori[1], curr_ori[2], -1]
        
    def get_current_action(self):
        return self.action

    def _get_move_trajectory(self, target_pos, obs, scene):
        eef_pos = obs['robot0_eef_pos']
        eef_ori = obs['robot0_eef_quat']
        in_hand = None
        for idx, obj in enumerate(scene):
            if obj['in_hand']:
                in_hand = idx
        if in_hand is not None:
            bbox_in_hand = scene[in_hand]['bbox']
            # obj_clearance = np.abs(
            #     obs['robot0_eef_pos'] - min([bbox_in_hand[i][2] for i in range(8)])
            # )
            obj_clearance = np.abs(
                obs['robot0_eef_pos'][2] - np.amin(bbox_in_hand[:, 2])
            )
        else:
            obj_clearance = 0.0

        last_pos = np.array(
            [
                target_pos[0],
                target_pos[1],
                target_pos[2]
            ]
        )
        travel_height = target_pos[2]
        clearance_height = self._get_max_height_in_line(eef_pos, target_pos, scene, exclude=in_hand)
        clearance_height = clearance_height + obj_clearance + 0.05
        if clearance_height > travel_height:
            travel_height = clearance_height
            target_pos[2] = clearance_height
        intermediate_pos = np.array(
            [
                eef_pos[0],
                eef_pos[1],
                travel_height
            ]
        )
        move_trajectory = [
            (intermediate_pos, eef_ori),
            (target_pos, eef_ori),
            (last_pos, eef_ori),
        ]
        # return [(target_pos + eef_pos) / 2 + np.array([0, 0.1, 0.1]), target_pos]
        return move_trajectory

    def _get_max_height_in_line(self, pos1, pos2, scene, exclude=None):
        path = Vector(
            Point(pos1[0], pos1[1]),
            Point(pos2[0], pos2[1])
        )
        travel_height = 0.0
        for idx, obj in enumerate(scene):
            if exclude is not None and exclude == idx:
                continue
            obj_z_dir = T.quat2mat(obj['ori'])[:, 2]
            # PRINT HERE
            # print(obj_z_dir)
            if self._angle_between(obj_z_dir, np.array([0, 0, 1])) < self.angle_up_tolerance:
                rec = Rectangle(
                    Point(obj['bbox'][0][0], obj['bbox'][0][1]),
                    Point(obj['bbox'][1][0], obj['bbox'][1][1]),
                    Point(obj['bbox'][2][0], obj['bbox'][2][1]),
                    Point(obj['bbox'][3][0], obj['bbox'][3][1])
                )
            else:
                rec1 = Rectangle(
                    Point(obj['bbox'][0][0], obj['bbox'][0][1]),
                    Point(obj['bbox'][1][0], obj['bbox'][1][1]),
                    Point(obj['bbox'][5][0], obj['bbox'][5][1]),
                    Point(obj['bbox'][4][0], obj['bbox'][4][1])
                )
                rec2 = Rectangle(
                    Point(obj['bbox'][0][0], obj['bbox'][0][1]),
                    Point(obj['bbox'][4][0], obj['bbox'][4][1]),
                    Point(obj['bbox'][7][0], obj['bbox'][7][1]),
                    Point(obj['bbox'][3][0], obj['bbox'][3][1])
                )
                if rec1.area() > rec2.area():
                    rec = rec1
                else:
                    rec = rec2
            # height = max([obj['bbox'][i][2] for i in range(8)])
            height = np.amax(obj['bbox'][:, 2])
            if rec.intersects(path):
                if height > travel_height:
                    travel_height = height
        return travel_height
    
    def _set_approach_grasp(self, target, obs, scene):
        target_name = scene[target]['name']
        if 'glass' in target_name:
            self._set_approach_cylinder(target, obs, scene)
        elif 'soda can' in target_name:
            self._set_approach_cylinder(target, obs, scene)
        elif 'thermos' in target_name:
            self._set_approach_cylinder(target, obs, scene)
        elif 'bowl' in target_name:
            self._set_approach_rant(target, obs, scene)
        elif 'mug' in target_name:
            self._set_approach_cylinder(target, obs, scene)
        elif 'pot' in target_name:
            self._set_approach_rant(target, obs, scene)
        elif 'pan' in target_name:
            self._set_approach_top_handle(target, obs, scene)
    
    def _set_approach_cylinder(self, target, obs, scene):
        self.move_dict = {}
        obj_ori = scene[target]['ori']
        # Check standing vs lying
        obj_z_dir = T.quat2mat(obj_ori)[:, 2]
        angle_to_up = self._angle_between(obj_z_dir, np.array([0, 0, 1]))
        if angle_to_up > self.angle_up_tolerance:
            # Lying
            # First gripper to upward position
            target_quat = self._get_closest_downward(obs[f'robot0_eef_quat'])
            self.move_dict['trajectory'] = []
            in_between_quat = self._get_quat_interp(
                obs[f'robot0_eef_quat'],
                target_quat,
                angle = self.interp_angle
            )
            for q in in_between_quat:
                self.move_dict['trajectory'].append((
                    obs['robot0_eef_pos'], q.copy()
                ))
            
            # Target orientation
            gripper_x_dir = obj_z_dir.copy()
            angle_to_floor = np.abs(90 - np.abs(np.rad2deg(angle_to_up)))
            gripper_x_on_floor = np.array(
                [gripper_x_dir[0], 
                gripper_x_dir[1],
                0]
            )
            gripper_x_on_floor = self._unit_vector(gripper_x_on_floor)
            if angle_to_floor < 20:
                # Keep gripper vertical
                target_x = gripper_x_on_floor
            else:
                target_x = gripper_x_dir
            target_y = np.cross(np.array([0, 0, -1]), gripper_x_on_floor)
            target_y = self._unit_vector(target_y)
            target_z = np.cross(target_x, target_y)
            target_z = self._unit_vector(target_z)
            target_mat = np.zeros((3, 3))
            target_mat[:, 0] = target_x
            target_mat[:, 1] = target_y
            target_mat[:, 2] = target_z
            target_quat = T.mat2quat(target_mat)

            in_between_quat = self._get_quat_interp(
                obs[f'robot0_eef_quat'],
                target_quat,
                angle = self.interp_angle
            )
            for q in in_between_quat:
                self.move_dict['trajectory'].append((
                    obs['robot0_eef_pos'], q.copy()
                ))

            self.move_dict['trajectory'].append(
                (obs['robot0_eef_pos'], target_quat.copy())
            )            

            # Target position with vert offset
            # Start with bbox centre
            bbox = scene[target]['bbox']
            target_pos = np.mean(bbox, 0)
            target_pos[2] = target_pos[2] - self.gripper_pad_height
            target_pos[2] = max(target_pos[2], 0.005)

            target_to_gripper_vec = -target_z
            t = (obs['robot0_eef_pos'][2] - target_pos[2]) / target_to_gripper_vec[2]
            init_pos = target_pos + target_to_gripper_vec * t
            diff = target_pos - init_pos
            for i in range(3):
                self.move_dict['trajectory'].append(
                    (init_pos + (i * diff) / 3, target_quat.copy())
                )            
            self.move_dict['trajectory'].append(
                (target_pos, target_quat.copy())
            )            
        else:
            # Standing
            # First gripper to upward position
            target_quat = self._get_closest_downward(obs[f'robot0_eef_quat'])
            self.move_dict['trajectory'] = []
            in_between_quat = self._get_quat_interp(
                obs[f'robot0_eef_quat'],
                target_quat,
                angle = self.interp_angle
            )
            for q in in_between_quat:
                self.move_dict['trajectory'].append((
                    obs['robot0_eef_pos'], q.copy()
                ))
            bbox = scene[target]['bbox']
            y_size = np.linalg.norm((bbox[0] - bbox[3]))
            x_size = np.linalg.norm((bbox[0] - bbox[1]))
            target_pos = scene[target]['pos']
            if np.abs(x_size - y_size) > 0.005:
                obj_mat = T.quat2mat(obj_ori)
                obj_y_dir = obj_mat[:, 1]
                obj_x_dir = obj_mat[:, 0]
                gripper_mat = T.quat2mat(obs[f'robot0_eef_quat'])
                gripper_mat = T.quat2mat(target_quat)
                gripper_y_dir = gripper_mat[:, 1]
                angle_between = self._angle_between(gripper_y_dir, obj_y_dir)
                if angle_between > np.pi:
                    angle_between = 2 * np.pi - angle_between
                if angle_between > np.pi / 2:
                    target_y = - obj_y_dir
                else:
                    target_y = obj_y_dir
                target_y = self._unit_vector(target_y)
                target_z = np.array([0, 0, -1])
                target_x = np.cross(target_y, target_z)
                target_x = self._unit_vector(target_x)
                target_mat = np.zeros((3, 3))
                target_mat[:, 0] = target_x
                target_mat[:, 1] = target_y
                target_mat[:, 2] = target_z
                target_quat = T.mat2quat(target_mat)
                target_pos = target_pos + (x_size - y_size) * obj_x_dir / 2

                in_between_quat = self._get_quat_interp(
                    obs[f'robot0_eef_quat'],
                    target_quat,
                    angle = self.interp_angle
                )
                for q in in_between_quat:
                    self.move_dict['trajectory'].append((
                        obs['robot0_eef_pos'], q.copy()
                    ))

                self.move_dict['trajectory'].append((
                    obs['robot0_eef_pos'], target_quat.copy()
                ))
            z_coord = bbox[:, 2]
            z_coord = np.max(z_coord) + self.move_clearance
            target_pos[2] = z_coord

            interp_pos = self._get_dist_interp(obs['robot0_eef_pos'], target_pos, self.interp_dist)
            for p in interp_pos:
                self.move_dict['trajectory'].append((
                p.copy(), target_quat.copy()
            ))

            self.move_dict['trajectory'].append((
                target_pos.copy(), target_quat.copy()
            ))
            # print()
            target_pos[2] = max(np.max(bbox[:, 2]) - self.grasp_depth, 0.005)
            interp_pos = self._get_dist_interp(obs['robot0_eef_pos'], target_pos, self.interp_dist)
            for p in interp_pos:
                self.move_dict['trajectory'].append((
                p.copy(), target_quat.copy()
            ))

            self.move_dict['trajectory'].append((
                target_pos.copy(), target_quat.copy()
            ))
        self.move_dict['current_target'] = self.move_dict['trajectory'].pop(0)


    def _set_approach_rant(self, target, obs, scene):
        self.move_dict = {}
        obj_ori = scene[target]['ori']
        obj_pos = scene[target]['pos']
        # Check standing vs lying
        obj_mat = T.quat2mat(obj_ori)
        obj_z_dir = obj_mat[:, 2]
        angle_to_up = self._angle_between(obj_z_dir, np.array([0, 0, 1]))
        if angle_to_up > self.angle_up_tolerance:
            #TODO
           pass
        else:
            # Standing
            # First gripper to upward position
            target_quat = self._get_closest_downward(obs[f'robot0_eef_quat'])
            self.move_dict['trajectory'] = []
            in_between_quat = self._get_quat_interp(
                obs[f'robot0_eef_quat'],
                target_quat,
                angle = self.interp_angle
            )
            for q in in_between_quat:
                self.move_dict['trajectory'].append((
                    obs['robot0_eef_pos'], q.copy()
                ))
            bbox = scene[target]['bbox']
            y_size = np.linalg.norm((bbox[0] - bbox[3]))
            x_size = np.linalg.norm((bbox[0] - bbox[1]))
            if np.abs(y_size - x_size) > 0.005:
                # print('HANDLE')
                # Rotate gripper to avoid handle
                # Gripper y and object y have to be parallel
                obj_y_dir = obj_mat[:, 1]
                gripper_mat = T.quat2mat(obs[f'robot0_eef_quat'])
                gripper_y_dir = gripper_mat[:, 1]
                angle_between = self._angle_between(gripper_y_dir, obj_y_dir)
                if angle_between > np.pi:
                    angle_between = 2 * np.pi - angle_between
                if angle_between > np.pi / 2:
                    target_y = - obj_y_dir
                else:
                    target_y = obj_y_dir
                target_y = self._unit_vector(target_y)
                target_z = np.array([0, 0, -1])
                target_x = np.cross(target_y, target_z)
                target_x = self._unit_vector(target_x)
                target_mat = np.zeros((3, 3))
                target_mat[:, 0] = target_x
                target_mat[:, 1] = target_y
                target_mat[:, 2] = target_z
                target_quat = T.mat2quat(target_mat)

                in_between_quat = self._get_quat_interp(
                    obs[f'robot0_eef_quat'],
                    target_quat,
                    angle = self.interp_angle
                )
                for q in in_between_quat:
                    self.move_dict['trajectory'].append((
                        obs['robot0_eef_pos'], q.copy()
                    ))

                self.move_dict['trajectory'].append((
                    obs['robot0_eef_pos'], target_quat.copy()
                ))

                offset_from_middle = y_size / 2
                if self.gripper_open_width > 3 * y_size / 4:
                    offset_from_middle = self.gripper_open_width
                target_pos_c1 = obj_pos - target_y * offset_from_middle
                target_pos_c2 = obj_pos + target_y * offset_from_middle
                if obj_pos[0] > 0.55:
                    if target_pos_c1[0] > target_pos_c2[0]:
                        target_pos = target_pos_c2
                    else:
                        target_pos = target_pos_c2
                else:
                    if target_pos_c1[0] < target_pos_c2[0]:
                        target_pos = target_pos_c2
                    else:
                        target_pos = target_pos_c2
                z_size = np.linalg.norm((bbox[0] - bbox[4]))
                grasp_depth = self.grasp_depth
                grasp_depth = min(grasp_depth, 3 * z_size / 4)
                target_pos[2] = max(
                    np.max(bbox[:, 2]) - grasp_depth,
                    0.005
                )
                init_pos = np.array(
                    [target_pos[0],
                    target_pos[1],
                    obs['robot0_eef_pos'][2]]
                )

                interp_pos = self._get_dist_interp(obs['robot0_eef_pos'], init_pos, self.interp_dist)
                for p in interp_pos:
                    self.move_dict['trajectory'].append((
                    p.copy(), target_quat.copy()
                ))

                self.move_dict['trajectory'].append((
                    init_pos.copy(), target_quat.copy()
                ))

                interp_pos = self._get_dist_interp(init_pos, target_pos, self.interp_dist)
                for p in interp_pos:
                    self.move_dict['trajectory'].append((
                    p.copy(), target_quat.copy()
                ))

                self.move_dict['trajectory'].append((
                    target_pos.copy(), target_quat.copy()
                ))

            else:
                # Grasp anywhere - cause symmetrical
                # Grasp without turning
                target_mat = T.quat2mat(target_quat)
                target_y = target_mat[:, 1]
                offset_from_middle = y_size / 2
                if self.gripper_open_width > 3 * y_size / 4:
                    offset_from_middle = self.gripper_open_width
                target_pos_c1 = obj_pos - target_y * offset_from_middle
                target_pos_c2 = obj_pos + target_y * offset_from_middle
                if obj_pos[0] > 0.55:
                    if target_pos_c1[0] > target_pos_c2[0]:
                        target_pos = target_pos_c2
                    else:
                        target_pos = target_pos_c2
                else:
                    if target_pos_c1[0] < target_pos_c2[0]:
                        target_pos = target_pos_c2
                    else:
                        target_pos = target_pos_c2
                z_size = np.linalg.norm((bbox[0] - bbox[4]))
                grasp_depth = self.grasp_depth
                grasp_depth = min(grasp_depth, 3 * z_size / 4)
                target_pos[2] = max(
                    np.max(bbox[:, 2]) - grasp_depth,
                    0.005
                )
                init_pos = np.array(
                    [target_pos[0],
                    target_pos[1],
                    obs['robot0_eef_pos'][2]]
                )
                in_between_quat = self._get_quat_interp(
                    obs[f'robot0_eef_quat'],
                    target_quat,
                    angle = self.interp_angle
                )
                for q in in_between_quat:
                    self.move_dict['trajectory'].append((
                        init_pos.copy(), q.copy()
                    ))
                self.move_dict['trajectory'].append((
                    init_pos.copy(), target_quat.copy()
                ))
                self.move_dict['trajectory'].append((
                    target_pos.copy(), target_quat.copy()
                ))

        self.move_dict['current_target'] = self.move_dict['trajectory'].pop(0)

    def _set_approach_top_handle(self, target, obs, scene):
        self.move_dict = {}
        obj_ori = scene[target]['ori']
        obj_pos = scene[target]['pos']
        # Check standing vs lying
        obj_mat = T.quat2mat(obj_ori)
        obj_z_dir = obj_mat[:, 2]
        angle_to_up = self._angle_between(obj_z_dir, np.array([0, 0, 1]))
        if angle_to_up > self.angle_up_tolerance:
           pass
        else:
            # Standing
            # First gripper to upward position
            target_quat = self._get_closest_downward(obs[f'robot0_eef_quat'])
            self.move_dict['trajectory'] = []
            in_between_quat = self._get_quat_interp(
                obs[f'robot0_eef_quat'],
                target_quat,
                angle = self.interp_angle
            )
            for q in in_between_quat:
                self.move_dict['trajectory'].append((
                    obs['robot0_eef_pos'], q.copy()
                ))
            bbox = scene[target]['bbox']
            x_size = np.linalg.norm((bbox[0] - bbox[1]))
            # Gripper x and object x have to be parallel
            obj_x_dir = obj_mat[:, 0]
            gripper_mat = T.quat2mat(obs[f'robot0_eef_quat'])
            gripper_x_dir = gripper_mat[:, 0]
            angle_between = self._angle_between(gripper_x_dir, obj_x_dir)
            if angle_between > np.pi:
                angle_between = 2 * np.pi - angle_between
            if angle_between > np.pi / 2:
                target_x = - obj_x_dir
            else:
                target_x = obj_x_dir
            target_x = self._unit_vector(target_x)
            target_z = np.array([0, 0, -1])
            target_y = np.cross(target_z, target_x)
            target_y = self._unit_vector(target_y)
            target_mat = np.zeros((3, 3))
            target_mat[:, 0] = target_x
            target_mat[:, 1] = target_y
            target_mat[:, 2] = target_z
            target_quat = T.mat2quat(target_mat)

            in_between_quat = self._get_quat_interp(
                obs[f'robot0_eef_quat'],
                target_quat,
                angle = self.interp_angle
            )
            for q in in_between_quat:
                self.move_dict['trajectory'].append((
                    obs['robot0_eef_pos'], q.copy()
                ))

            self.move_dict['trajectory'].append((
                obs['robot0_eef_pos'], target_quat.copy()
            ))

            offset_from_middle = 3 * x_size / 8
            target_pos = obj_pos - target_x * offset_from_middle
            z_size = np.linalg.norm((bbox[0] - bbox[4]))
            grasp_depth = self.grasp_depth
            grasp_depth = min(grasp_depth, 3 * z_size / 4)
            target_pos[2] = max(
                np.max(bbox[:, 2]) - grasp_depth,
                0.005
            )
            init_pos = np.array(
                [target_pos[0],
                target_pos[1],
                obs['robot0_eef_pos'][2]]
            )
            self.move_dict['trajectory'].append((
                init_pos.copy(), target_quat.copy()
            ))
            self.move_dict['trajectory'].append((
                target_pos.copy(), target_quat.copy()
            ))

        self.move_dict['current_target'] = self.move_dict['trajectory'].pop(0)

    def _unit_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def _angle_between(self, v1, v2):
        v1_u = self._unit_vector(v1)
        v2_u = self._unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def _get_rotation_from_vec(self, rot_vec, angle_diff):
        c = np.cos(angle_diff)
        s = np.sin(angle_diff)
        ux = rot_vec[0]
        uy = rot_vec[1]
        uz = rot_vec[2]
        return np.array([
            [
                c + ux * ux * (1 - c),
                ux * uy * (1 - c) - uz * s,
                ux * uz * (1 - c) + uy * s
            ],
            [
                uy * ux * (1 - c) + uz * s,
                c + uy * uy * (1 - c),
                uy * uz * (1 - c) - ux * s
            ],
            [
                uz * ux * (1 - c) - uy * s,
                uz * uy * (1 - c) + ux * s,
                c + uz * uz * (1 - c)
            ]
        ])

    def _get_closest_downward(self, quat):
        gripper_mat = T.quat2mat(quat)
        z_dir = gripper_mat[:, 2]
        z_targ = np.array([0, 0, -1])
        angle_diff = self._angle_between(z_dir, z_targ)
        if angle_diff < self.angle_diff_tol:
            target_quat = quat
        else:
            y_targ = np.array([gripper_mat[0, 1], gripper_mat[1, 1], 0])
            y_targ = self._unit_vector(y_targ)
            x_targ = np.cross(y_targ, z_targ)
            gripper_mat[:, 0] = x_targ
            gripper_mat[:, 1] = y_targ
            gripper_mat[:, 2] = z_targ
            target_quat = T.mat2quat(gripper_mat)
        return target_quat

    def _get_quat_interp(self, start, end, angle=np.pi/4):
        quats = []
        quat_dist = T.quat_distance(end, start)
        angle_between = T.quat2axisangle(quat_dist)
        angle_corr = []
        # print([a * 180 / np.pi for a in angle_between])
        for a in angle_between:
            new_ang = a
            if new_ang > np.pi:
                new_ang = new_ang - 2 * np.pi
            elif new_ang < -np.pi:
                new_ang = new_ang + 2 * np.pi
            angle_corr.append(new_ang)
        angle_between = angle_corr
        # print([a * 180 / np.pi for a in angle_between])
        angle_between = np.linalg.norm(angle_between)
        if angle_between > np.pi:
            angle_between = angle_between - np.pi
        angle_between = np.abs(angle_between)
        # print("KJASHDKJSAHKDSAHKDHAKJSHK")
        # print(angle_between * 180 / np.pi)
        frac = np.ceil(angle_between / angle)
        # frac = min([frac, 80])
        for i in range(1, int(frac)):
            quats.append(
                T.quat_slerp(start, end, i / frac)
            )
        return quats

    def _get_dist_interp(self, start, end, interp_dist=0.01):
        positions = []
        direction = end - start
        dist = np.linalg.norm(direction)
        direction = direction / dist
        frac = np.ceil(dist / interp_dist)
        incr = dist / frac
        for i in range(1, int(frac)):
            positions.append(start + i * incr * direction)
        return positions

    def _get_euler_diff_between_quat(self, quat_from, quat_to):
        # mat_from = T.quat2mat(quat_from)
        # print(mat_from)
        # mat_to = T.quat2mat(quat_to)
        # print(mat_to)
        # mat_from_to = np.matmul(mat_from.T, mat_to)
        # print(mat_from_to)
        src = self._get_euler_from_quat(quat_from)
        dst = self._get_euler_from_quat(quat_to)
        diff = dst - src
        # print(diff)
        # print(self._get_euler_from_quat_zyx(quat_to) - self._get_euler_from_quat_zyx(quat_from))
        diff_corr = np.zeros_like(diff)
        for i in range(diff.shape[0]):
            d = diff[i]
            if d > 2 * np.pi:
                # print('>2pi', d)
                factor = np.floor(d / 2 / np.pi)
                d = d - factor * 2 * np.pi
                # print('>2pi', d)
            if d < - 2 * np.pi:
                # print('<-2pi', d)
                factor = np.floor(- d / 2 / np.pi)
                d = d + factor * 2 * np.pi
                # print('<-2pi', d)
            if d > np.pi:
                # print('>pi', d)
                d = d - 2 * np.pi
                # print('>pi', d)
            elif d < - np.pi:
                # print('<-pi', d)
                d = d + 2 * np.pi
                # print('<-pi', d)
            diff_corr[i] = d    

        return diff_corr

    def _get_euler_from_quat(self, quat):
        x = quat[0]
        y = quat[1]
        z = quat[2]
        w = quat[3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        return np.array([roll_x, pitch_y, yaw_z])

    def _get_euler_from_quat_zyx(self, quat):
        x = quat[0]
        y = quat[1]
        z = quat[2]
        w = quat[3]
        eul = [
            np.arctan2( 2*(x*y+w*z), w**2 + x**2 - y**2 - z**2 ),
            np.arcsin( -2*(x*z-w*y) ),
            np.arctan2( 2*(y*z+w*x), w**2 - x**2 - y**2 + z**2 )]
        return np.array(eul)

    def _euler_to_quat(self, euler):
        x = euler[0]
        y = euler[1]
        z = euler[2]

class Point:
    def __init__(self, x, y, z=None) -> None:
        self.x = x
        self.y = y
        self.z = z

    def is_2d(self):
        return (self.z is None)

    def __str__(self):
        if self.is_2d():
            return f"Point: ({self.x:02f}, {self.y:02f})"
        return f"Point: ({self.x:02f}, {self.y:02f}, {self.z:02f})"

class Vector:
    def __init__(self, p1, p2) -> None:
        self.ps = p1
        self.pe = p2

    def intersects(self, vec):
        if vec.is_2d():
            assert self.is_2d()
            o1 = self._orientation(self.ps, self.pe, vec.ps)
            o2 = self._orientation(self.ps, self.pe, vec.pe)
            o3 = self._orientation(vec.ps, vec.pe, self.ps)
            o4 = self._orientation(vec.ps, vec.pe, self.pe)
        
            # General case
            if ((o1 != o2) and (o3 != o4)):
                return True
        
            # Special Cases
        
            # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
            if ((o1 == 0) and self._on_segment(self.ps, vec.ps, self.pe)):
                return True
        
            # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
            if ((o2 == 0) and self._on_segment(self.ps, vec.pe, self.pe)):
                return True
        
            # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
            if ((o3 == 0) and self._on_segment(vec.ps, self.ps, vec.pe)):
                return True
        
            # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
            if ((o4 == 0) and self._on_segment(vec.ps, self.pe, vec.pe)):
                return True
        
            # If none of the cases
            return False
        else:
            assert not self.is_2d()
            raise NotImplementedError()

    def _on_segment(self, p, q, r):
        if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
            (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
            return True
        return False
    
    def _orientation(self, p, q, r):
        # to find the orientation of an ordered triplet (p,q,r)
        # function returns the following values:
        # 0 : Collinear points
        # 1 : Clockwise points
        # 2 : Counterclockwise
        
        # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/        
        val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
        if (val > 0):
            # Clockwise orientation
            return 1
        elif (val < 0):
            # Counterclockwise orientation
            return 2
        else:
            # Collinear orientation
            return 0
    
    def _orientation_self(self, r):
        # to find the orientation of an ordered triplet (p,q,r)
        # function returns the following values:
        # 0 : Collinear points
        # 1 : Clockwise points
        # 2 : Counterclockwise
        
        p = self.ps
        q = self.pe
        return self._orientation(p, q, r)

    def is_2d(self):
        return self.ps.is_2d()

class Rectangle:
    def __init__(self, p1, p2, p3, p4) -> None:
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.sides = [
            Vector(p1, p2),
            Vector(p2, p3),
            Vector(p3, p4),
            Vector(p4, p1),
        ]

    def intersects(self, vec):
        for side in self.sides:
            if side.intersects(vec):
                return True
        return False

    def contains(self, point):
        orientations = []
        for side in self.sides:
            orientations.append(
                side._orientation_self(point)
            )
        if 0 in orientations:
            return True
        if len(set(orientations)) == 1:
            return True
        return False

    def area(self):
        area = 0.0
        area += self.p1.x * self.p2.y
        area -= self.p2.x * self.p1.y
        area += self.p2.x * self.p3.y
        area -= self.p3.x * self.p2.y
        area += self.p3.x * self.p4.y
        area -= self.p4.x * self.p3.y
        area += self.p4.x * self.p1.y
        area -= self.p1.x * self.p4.y
        area = np.abs(area) / 2.0
        return area

    def mid_point(self):
        x_sum = 0
        y_sum = 0
        # z_sum = 0
        for side in self.sides:
            p = side.ps
            x_sum += p.x
            y_sum += p.y
        x_sum /= 4
        y_sum /= 4
        return Point(x_sum, y_sum)

    def dist_to_point(self, p):
        min_dist = None
        # print(min_dist)
        for side in self.sides:
            # dists = [
            #     np.sqrt((side.ps.x - p.x) ** 2 + (side.ps.y - p.y) ** 2),
            #     np.sqrt((side.pe.x - p.x) ** 2 + (side.pe.y - p.y) ** 2),
            # ]
            # dists.append(
            #     np.abs(
            #         (side.pe.x - side.ps.x) * (side.ps.y - p.y) - (side.ps.x - p.x) * (side.pe.y - side.ps.y)
            #     ) / np.sqrt(
            #         (side.pe.x - side.ps.x) ** 2 + (side.pe.y - side.ps.y) ** 2 
            #     )
            # )
            # side_dist = np.amin(np.array(dists))
            # print(dists, side_dist)
            A = p.x - side.ps.x
            B = p.y - side.ps.y
            C = side.pe.x - side.ps.x
            D = side.pe.y - side.ps.y
            dot = A * C + B * D
            len_sq = C * C + D * D

            param = -1 
            if len_sq > 0:
                param = dot / len_sq

            if param < 0:
                xx = side.ps.x
                yy = side.ps.y
            elif param > 1:
                xx = side.pe.x
                yy = side.pe.y
            else:
                xx = side.ps.x + param * C
                yy = side.ps.y + param * D

            dx = p.x - xx
            dy = p.y - yy

            side_dist = np.sqrt(dx * dx  + dy * dy)

            if min_dist is None:
                min_dist = side_dist
            else:
                if side_dist < min_dist:
                    min_dist = side_dist
        return min_dist