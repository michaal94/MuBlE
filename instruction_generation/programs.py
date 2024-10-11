def generate_programs(prog_template, scene_struct, descr_dict, property_names):
    prog_details = []
    idxs = []
    # print(prog_list)
    for sub_prog in prog_template:
        # print(sub_prog)
        if sub_prog['input_values'] is not None:
            if sub_prog['input_values'] in descr_dict:
                descr = descr_dict[sub_prog['input_values']]
            else:
                descr = sub_prog['input_values']
        else:
            descr = None
        if sub_prog['type'] == "scene":
            out_seq, idxs = scene(scene_struct, idxs, descr)
        if sub_prog['type'] == "filter_object":
            out_seq, idxs = filter_object_gen(scene_struct, idxs, descr, property_names)
        if sub_prog['type'] == "filter_relate":
            out_seq, idxs = filter_relate_gen(scene_struct, idxs, descr)
        if sub_prog['type'] == "query_weight":
            out_seq, idxs = query_weight(scene_struct, idxs, descr)
        if sub_prog['type'] == "filter_weight":
            out_seq, idxs = filter_weight(scene_struct, idxs, descr)
        if sub_prog['type'] == "filter_table_part":
            out_seq, idxs = filter_table_part(scene_struct, idxs, descr)
        if sub_prog['type'] == "order_weight":
            out_seq, idxs = order_weight(scene_struct, idxs, descr)
        if sub_prog['type'] == "pick_up":
            out_seq, idxs = pick_up(scene_struct, idxs, descr)
        if sub_prog['type'] == "move_to":
            out_seq, idxs = move_to(scene_struct, idxs, descr)
        if sub_prog['type'] == "move_opposite":
            out_seq, idxs = move_opposite(scene_struct, idxs, descr)
        if sub_prog['type'] == "stack":
            out_seq, idxs = stack(scene_struct, idxs, descr, prog_details)
        if out_seq is None:
            return None
        prog_details += out_seq
    return prog_details

def filter_table_part(scene_struct, idxs, descr):
    seq_struct = {
        "type": "filter_table_part",
        "input": idxs,
        "input_value": descr 
    }
    out_idxs = []
    for idx in idxs:
        if descr in scene_struct['objects'][idx]['table_part']:
            out_idxs.append(idx)
    seq_struct["output"] = out_idxs
    return [seq_struct], out_idxs

def move_opposite(scene_struct, idxs, descr):
    tp = get_opposite_table_part(descr)
    seq_struct = {
        "type": "move_to",
        "input": idxs,
        "input_value": tp,
        "output": idxs
    }
    return [seq_struct], idxs

def get_opposite_table_part(tp):
    if tp == 'left':
        return 'right'
    elif tp == 'right':
        return 'left'
    elif tp == 'front':
        return 'back'
    elif tp == 'back':
        return 'front'
    else:
        raise NotImplementedError()

def move_to(scene_struct, idxs, descr):
    seq_struct = {
        "type": "move_to",
        "input": idxs,
        "input_value": descr,
        "output": idxs
    }
    return [seq_struct], idxs

def order_weight(scene_struct, idxs, descr):
    seq_struct = {
        "type": "order_weight",
        "input": idxs,
        "input_value": descr 
    }
    weights = {}
    for idx in idxs:
        weights[idx] = scene_struct['objects'][idx]['weight_gt']
    
    reverse = (descr == "descending")
    out_idx = [k for k, _ in sorted(weights.items(), key=lambda item: item[1], reverse=reverse)]

    if not scene_struct['objects'][out_idx[0]]['stackable']:
        return None, None
    if not scene_struct['objects'][out_idx[-1]]['stack_base']:
        return None, None
    for i in range(1, len(out_idx) - 1):
        if not scene_struct['objects'][out_idx[i]]['stackable']:
            return None, None
        if not scene_struct['objects'][out_idx[i]]['stack_base']:
            return None, None

    # if len(out_idx) > 3:
    #     print('asdfasdf')

    seq_struct["output"] = out_idx
    return [seq_struct], out_idx

def stack(scene_struct, idxs, descr, out_seq):
    outs_idxs = []
    for i in range(1, len(out_seq)):
        if out_seq[i]['type'] == 'scene':
            outs_idxs.append(i - 1)
    outs_idxs.append(len(out_seq) - 1)
    out_idxs = []
    for oidx in outs_idxs:
        out_idxs.append(
            out_seq[oidx]['output']
        )
    seq_struct = {
        "input": None,
        "type": "stack",
        "input_value": None,
        "output": out_idxs
    }
    return [seq_struct], out_idxs

def scene(scene_struct, idxs, descr):
    out_idxs = list(range(len(scene_struct['objects'])))
    seq_struct = {
        "input": None,
        "type": "scene",
        "input_value": None,
        "output": out_idxs
    }
    return [seq_struct], out_idxs

def filter_weight(scene_struct, idxs, descr):
    seq_struct = {
        "type": "filter_weight",
        "input": idxs,
        "input_value": descr 
    }
    weights = {}
    for idx in idxs:
        weights[idx] = scene_struct['objects'][idx]['weight_gt']
    if descr == 'lightest':
        out_idx = min(weights, key=weights.get)
    elif descr == 'heaviest':
        out_idx = max(weights, key=weights.get)
    seq_struct["output"] = [out_idx]
    return [seq_struct], [out_idx]

def query_weight(scene_struct, idxs, descr):
    seq_struct = {
        "type": "query_weight",
        "input": idxs,
        "input_value": None 
    }
    out_list = []
    for idx in idxs:
        out_list.append(scene_struct['objects'][idx]['weight_gt'])
    seq_struct["output"] = out_list
    return [seq_struct], idxs

def pick_up(scene_struct, idxs, descr):
    seq_struct = {
        "type": "pick_up",
        "input": idxs,
        "input_value": None,
        "output": idxs
    }
    return [seq_struct], idxs

def filter_relate_gen(scene_struct, idxs, descr):
    seq_struct = {
        "type": "filter_relate",
        "input": idxs,
        "input_value": descr
    }
    if len(idxs) != 1:
        seq_struct["output"] = []
        return seq_struct, []
    obj_idx = idxs[0]
    out_idxs = scene_struct['objects'][obj_idx]['directions'][descr]
    seq_struct["output"] = out_idxs
    return [seq_struct], out_idxs


def filter_object_gen(scene_struct, inp_idxs, descr, property_names):
    out_prog_seq = []
    out_idxs = inp_idxs
    # print(descr)
    source_obj_idx = descr[0]
    for prop_idx in descr[1]:
        seq_element = {}
        if out_idxs == "<SCENE>":
            out_idxs = list(range(len(scene_struct['objects'])))
        prop_name = property_names[prop_idx]
        prop_val = scene_struct['objects'][source_obj_idx][prop_name]
        out_idxs = filter_handler(scene_struct, out_idxs, prop_name, prop_val)
        seq_element["type"] = "filter_" + prop_name
        seq_element["input"] = out_idxs
        seq_element["input_value"] = prop_val
        seq_element["output"] = out_idxs
        out_prog_seq.append(seq_element)
        # print(out_idxs)
    # print(out_prog_seq)
    return out_prog_seq, out_idxs


def filter_handler(scene_struct, inp_idxs, prop_name, prop_val):
    out_idxs = []
    for i, obj in enumerate(scene_struct['objects']):
        if i in inp_idxs:
            if obj[prop_name] == prop_val:
                out_idxs.append(i)
    return out_idxs


def generate_task(task, prog_details, subs_dict):
    task_details = []

    def find_sub(tag):
        out_val = tag
        if tag in subs_dict:
            out_val = subs_dict[tag]
        if tag in prog_details:
            out_val = prog_details[tag][-1]["output"]
        return out_val

    for subtask in task:
        out_task = {}
        for key, value in subtask.items():
            # print(value)
            new_val = value
            if isinstance(value, list):
                new_val = []
                for val in value:
                    new_val += find_sub(val)
            else:
                new_val = find_sub(value)
            out_task[key] = new_val
        task_details.append(out_task)

    return task_details
