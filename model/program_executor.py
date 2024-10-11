class ProgramStatus:
    SUCCESS = 0
    FAILURE = 1
    ACTION = 2
    FINAL_ACTION = 3
    
class ProgramExecutor:
    def __init__(self, return_single_programs=True) -> None:
        self.return_single_programs = return_single_programs
        self.scene = None

    def execute(self, scene, program_list):
        self._set_scene(scene)
        assert program_list[0] == 'scene', "Program must start with scene call"
        # Collect series of filtered outputs on a stack
        # E.g. for stacking plastic obj on ceramic obj
        # scene -> filter_material[plastic] -> scene -> filter_material[ceramic] -> stack
        # scene -> filter_material[plastic] goes to stack[0]
        # scene -> filter_material[ceramic] goes to stack[1]
        # then stack calls actionable with the order of stack list
        stack = []
        for i, program in enumerate(program_list):
            if len(stack) > 0:
                if len(stack[-1]) < 1:
                    return {
                        'STATUS': ProgramStatus.FAILURE,
                        'ACTION': None,
                        'ANSWER': None
                    }
            if self._is_actionable(program):
                if i == (len(program_list) - 1):
                    last_action = True
                else:
                    last_action = False
                if 'query' in program:
                    last_action = False
                if 'filter' in program:
                    output, action_required = self._call_filter_actionable(stack[-1], program)
                elif 'move' in program:
                    output, action_required = self._call_move_actionable(stack[-1], program)
                elif 'query' in program:
                    output, action_required = self._call_query_actionable(stack[-1], program)
                elif 'order_weight' in program:
                    output, action_required = self._call_order_weight_actionable(stack[-1], program)
                elif 'stack' in program:
                    output, action_required = self._call_stack_actionable(stack, program)
                else:
                    output, action_required = self._call_generic_actionable(stack, program)
                # print(output)
                if action_required:
                    if last_action:
                        return {
                            'STATUS': ProgramStatus.FINAL_ACTION,
                            'ACTION': output,
                            'ANSWER': None
                        }
                    else:
                        return {
                            'STATUS': ProgramStatus.ACTION,
                            'ACTION': output,
                            'ANSWER': None
                        }
                else:
                    stack[-1] = output
            else:
                if 'scene' in program:
                    output = self._call_scene()
                    stack.append(output)
                if 'filter' in program:
                    output = self._call_filter_nonaction(stack[-1], program)
                    stack[-1] = output
        self._reset_scene()
        return {
            'STATUS': ProgramStatus.SUCCESS,
            'ACTION': None,
            'ANSWER': stack
        }

    def _is_actionable(self, function):
        if 'filter' in function:
            if 'weight' in function:
                return True
        if 'query' in function:
            if 'weight' in function:
                return True
        if 'pick_up' in function:
            return True
        if 'order_weight' in function:
            return True
        if 'stack' in function:
            return True
        if 'move' in function:
            return True
        return False

    def _set_scene(self, scene):
        if 'objects' in scene:
            self.scene = scene['objects']
        else:
            self.scene = scene

    def _reset_scene(self):
        self.scene = None

    def _call_scene(self):
        return list(range(len(self.scene)))
    
    def _call_filter_nonaction(self, inp, program):
        _, property_name, value = self._program_decompose(program)
        # print(property_name, value)
        output = []
        if property_name == 'table_part':
            assert value in ['front', 'back', 'left', 'right']
            assert value in ['left', 'right']
            for idx in inp:
                if value == "left":
                    if self.scene[idx]['pos'][1] < 0:
                        output.append(idx)
                elif value == "right":
                    if self.scene[idx]['pos'][1] > 0:
                        output.append(idx)
        else:
            for idx in inp:
                if self.scene[idx][property_name] == value:
                    output.append(idx)
        # print(output)
        return output

    def _call_generic_actionable(self, inp, program):
        return {
            'task': program,
            'target': inp
        }, True
    
    def _call_move_actionable(self, inp, program):
        task, property_name, value = self._program_decompose(program)
        return {
            'task': program,
            'target': [inp]
        }, True

    def _call_stack_actionable(self, inp, program):
        if len(inp[-1]) > 1:
            return {
                'task': program,
                'target': inp
            }, True
        else:
            inp_merged = []
            for inp_item in inp:
                inp_merged += inp_item
            return {
                'task': program,
                'target': [inp_merged]
            }, True  

    def _call_filter_actionable(self, inp, program):
        _, property_name, value = self._program_decompose(program)
        output = []
        value_list = []
        for idx in inp:
            value_list.append(self.scene[idx][property_name])
        if any([v is None for v in value_list]):
            missing_idxs = []
            for i, idx in enumerate(inp):
                if value_list[i] is None:
                    missing_idxs.append(idx)
                task = f'measure_{property_name}'
            if self.return_single_programs:
                missing_idxs = [missing_idxs[0]]
            return {
                'task': task,
                'target': [missing_idxs]
            }, True
        else:
            if value == 'smallest':
                minval = min(value_list)
                argmin = value_list.index(minval)
                output = [inp[argmin]]
            elif value == 'biggest':
                maxval = max(value_list)
                argmax = value_list.index(maxval)
                output = [inp[argmax]]
            else:
                raise NotImplementedError()
            return output, False

    def _call_query_actionable(self, inp, program):
        _, property_name, _ = self._program_decompose(program)
        output = []
        value_list = []
        for idx in inp:
            value_list.append(self.scene[idx][property_name])
        if any([v is None for v in value_list]):
            missing_idxs = []
            for i, idx in enumerate(inp):
                if value_list[i] is None:
                    missing_idxs.append(idx)
                task = f'measure_{property_name}'
            if self.return_single_programs:
                missing_idxs = [missing_idxs[0]]
            return {
                'task': task,
                'target': [missing_idxs]
            }, True
        else:
            output = value_list
            return output, False

    def _call_order_weight_actionable(self, inp, program):
        _, property_name, value = self._program_decompose(program)
        output = []
        value_list = []
        for idx in inp:
            value_list.append(self.scene[idx]['weight'])
        if any([v is None for v in value_list]):
            missing_idxs = []
            for i, idx in enumerate(inp):
                if value_list[i] is None:
                    missing_idxs.append(idx)
                task = f'measure_weight'
            if self.return_single_programs:
                missing_idxs = [missing_idxs[0]]
            return {
                'task': task,
                'target': [missing_idxs]
            }, True
        else:
            weights = {}
            for idx in inp:
                weights[idx] = self.scene[idx]['weight']
            reverse = (value == "descending")
            out_idx = [k for k, _ in sorted(weights.items(), key=lambda item: item[1], reverse=reverse)]
            output = [outid for outid in out_idx]
            return output, False

    def _program_decompose(self, program):
        if '[' in program:
            split_list = program.split('[')
            assert len(split_list) == 2
            function_name = split_list[0]
            value = split_list[1].strip(']')
            function_split = function_name.split('_')
            root = function_split[0]
            property_name = function_split[1]
            if len(function_split) > 2:
                property_name = f"{property_name}_{function_split[2]}"
            if value == 'heaviest':
                value = 'biggest'
            if value == 'lightest':
                value = 'smallest'
        else:
            function_name = program
            value = None
            function_split = function_name.split('_')
            root = function_split[0]
            property_name = function_split[1]
        return root, property_name, value