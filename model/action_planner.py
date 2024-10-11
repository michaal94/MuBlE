import abc
import numpy as np


class ActionPlanInference:
    def __init__(self, params={}) -> None:
        pass

    @abc.abstractmethod
    def get_action_sequence(self, goal, scene_state):
        '''
        Abstract class for the implementation of instruction -> symbolic program inference
        Expected input: dict: {'instruction': [...], ...}
        Expected output: list: [symbolic_function1, ...] 
        '''
        pass

class ActionGTPlanner(ActionPlanInference):
    def get_action_sequence(self, goal, scene_state):
        task = goal['task']
        target = goal['target']
        action_list = []
        subseq = self._get_subsequence(goal, target[0], scene_state)
        action_list = action_list + subseq
        return action_list

    def _get_subsequence(self, goal, target, scene_state):
        if goal['task'] == 'measure_weight' or goal['task'] == 'pick_up':
            target_idx = target[0]
            target_state = scene_state[target_idx]
            in_hand = None
            for idx, obj in enumerate(scene_state):
                if obj['in_hand']:
                    in_hand = idx
            if in_hand is not None:
                if in_hand != target_idx:
                    if scene_state[in_hand]['raised']:
                        action_list = [
                            ('put_down', in_hand),
                            ('release', None),
                            ('move', target_idx),
                            ('approach_grasp', target_idx),
                            ('grasp', target_idx),
                            ('pick_up', target_idx)
                        ]
                    else:
                        action_list = [
                            ('release', None),
                            ('move', target_idx),
                            ('approach_grasp', target_idx),
                            ('grasp', target_idx),
                            ('pick_up', target_idx)
                        ]
                else:
                    if target_state['raised']:
                        action_list = []
                    else:
                        action_list = [('pick_up', target_idx)]
            elif target_state['approached']:
                action_list = [
                    ('grasp', target_idx),
                    ('pick_up', target_idx)
                ]
            elif target_state['gripper_over']:
                action_list = [
                    ('approach_grasp', target_idx),
                    ('grasp', target_idx),
                    ('pick_up', target_idx)
                ]
            else:
                action_list = [
                    ('move', target_idx),
                    ('approach_grasp', target_idx),
                    ('grasp', target_idx),
                    ('pick_up', target_idx)
                ]
            return action_list
        elif goal['task'] == 'stack':
            in_hand = None
            for idx, obj in enumerate(scene_state):
                if obj['in_hand']:
                    in_hand = idx
            action_list = []
            for i in reversed(range(len(target) - 1)):
                top = target[i]
                bottom = target[i + 1]
                pos_top = scene_state[top]['pos']
                pos_bot = scene_state[bottom]['pos']
                if np.linalg.norm(pos_top[0:2] - pos_bot[0:2]) < 0.1:
                    if (pos_top[2] - pos_bot[2]) > 0.001:
                        if in_hand is None:
                            continue
                        else:
                            if in_hand != top:
                                continue
                            top_obj_bot = np.amin(scene_state[top]['bbox'][:, 2])
                            bot_obj_top = np.amax(scene_state[bottom]['bbox'][:, 2])
                            # if scene_state[top]['raised']:
                            if top_obj_bot - bot_obj_top > 0.05:
                                action_list = [
                                    ('put_down', in_hand),
                                    ('release', None)
                                ]
                            else:
                                action_list = [
                                    ('release', None)
                                ]
                            return action_list
                    else:
                        return []
                else:
                    if in_hand is not None:
                        if in_hand != top:
                            if scene_state[in_hand]['raised']:
                                action_list = [
                                    ('put_down', in_hand),
                                    ('release', None),
                                    ('move', top),
                                    ('approach_grasp', top),
                                    ('grasp', top),
                                    ('pick_up', top),
                                    ('move', bottom),
                                    ('put_down', top),
                                    ('release', None),
                                ]
                            else:
                                action_list = [
                                    ('release', None),
                                    ('move', top),
                                    ('approach_grasp', top),
                                    ('grasp', top),
                                    ('pick_up', top),
                                    ('move', bottom),
                                    ('put_down', top),
                                    ('release', None),
                                ]
                        else:
                            if scene_state[top]['raised']:
                                action_list = [
                                    ('move', bottom),
                                    ('put_down', top),
                                    ('release', None)
                                ]
                            else:
                                action_list = [
                                    ('pick_up', top),
                                    ('move', bottom),
                                    ('put_down', top),
                                    ('release', None)
                                ]
                    elif scene_state[top]['approached']:
                        action_list = [
                            ('grasp', top),
                            ('pick_up', top),
                            ('move', bottom),
                            ('put_down', top),
                            ('release', None)
                        ]
                    elif scene_state[top]['gripper_over']:
                        action_list = [
                            ('approach_grasp', top),
                            ('grasp', top),
                            ('pick_up', top),
                            ('move', bottom),
                            ('put_down', top),
                            ('release', None)
                        ]
                    else:
                        action_list = [
                            ('move', top),
                            ('approach_grasp', top),
                            ('grasp', top),
                            ('pick_up', top),
                            ('move', bottom),
                            ('put_down', top),
                            ('release', None)
                        ]
                    return action_list
            return action_list
        elif 'move' in goal['task']:
            side = goal['task'].split('[')[1].strip(']')
            in_hand = None
            for idx, obj in enumerate(scene_state):
                if obj['in_hand']:
                    in_hand = idx
            action_list = []
            if in_hand is not None:
                if in_hand not in target:
                    if scene_state[in_hand]['raised']:
                        action_list += [
                            ('put_down', in_hand),
                            ('release', None)
                        ]
                    else:
                        action_list += [
                            ('release', None)
                        ]
                else:
                    new_target_list = []
                    new_target_list.append(in_hand)
                    for idx in target:
                        if idx != in_hand:
                            new_target_list.append(idx)
                    target = new_target_list
    
            for idx in target:
                y_pos = scene_state[idx]['pos'][1]
                if (side == 'right' and y_pos > 0) or (side == 'left' and y_pos < 0):
                    if idx == in_hand:
                        if scene_state[in_hand]['raised']:
                            action_list += [
                                ('put_down', in_hand),
                                ('release', None)
                            ]
                        else:
                            action_list += [
                                ('release', None)
                            ]
                    else:
                        continue
                else:
                    if idx == in_hand:
                        if scene_state[idx]['raised']:
                            action_list += [
                                ('move', side),
                                ('put_down', in_hand),
                                ('release', None)
                            ]
                        else:
                            action_list += [
                                ('pick_up', idx),
                                ('move', side),
                                ('put_down', in_hand),
                                ('release', None)
                            ]
                    else:
                        if scene_state[idx]['approached']:
                            action_list += [   
                                ('grasp', idx),
                                ('pick_up', idx),
                                ('move', side),
                                ('put_down', idx),
                                ('release', None)
                            ]
                        elif scene_state[idx]['gripper_over']:
                            action_list += [   
                                ('approach_grasp', idx),
                                ('grasp', idx),
                                ('pick_up', idx),
                                ('move', side),
                                ('put_down', idx),
                                ('release', None)
                            ]
                        else:
                            action_list += [   
                                ('move', idx),
                                ('approach_grasp', idx),
                                ('grasp', idx),
                                ('pick_up', idx),
                                ('move', side),
                                ('put_down', idx),
                                ('release', None)
                            ]
            return action_list
    
        else:
            return []