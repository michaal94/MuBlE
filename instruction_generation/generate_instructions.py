import os
import re
import json
import copy
import time
import random
import argparse
import itertools
import programs
from config import Config
from constraints import Constraints
from instruction_collector import collect_instructions

parser = argparse.ArgumentParser()

# Config file
parser.add_argument(
    "--config", "-c",
    default=os.path.join("configs", "default.json")
)


def generate_instruction_from_template(scene_struct, template, metadata,
                                       synonyms, plurals, substitutions,
                                       num_instructions=1,
                                       extra_unique_steps=1,
                                       verbose=False):

    if "table_part" not in scene_struct['objects'][0]:
        for obj in scene_struct['objects']:
            obj['table_part'] = []
            if obj['3d_coords'][1] > 0:
                obj['table_part'].append('right')
            else:
                obj['table_part'].append('left')


    # Resolve constraints and reject some object combinations
    constraints = {}
    for constr in template['constraints']:
        if constr['type'] not in constraints:
            constraints[constr['type']] = [constr['target']]
        else:
            constraints[constr['type']].append(constr['target'])

    output_dict = []

    # print(template['text'])
    # First check some constraints for scene requirements
    if "SCENE_UNIQUE" in constraints.keys():
        scene_unique = Constraints.check_scene_unique(
            constraints["SCENE_UNIQUE"],
            scene_struct
        )
        if not scene_unique:
            if verbose:
                print("Template cannot be used on the scene, "
                      + "shared properties not fulfilled.")
            return output_dict

    if "SCENE_SHARE_PROP" in constraints.keys():
        scene_share_prop = Constraints.check_scene_prop_share(
            constraints["SCENE_SHARE_PROP"],
            scene_struct
        )
        if not scene_share_prop:
            if verbose:
                print("Template cannot be used on the scene, "
                      + "shared properties not fulfilled.")
            return output_dict

    # Create dictionary of parameter types
    parameters = {}
    for param in template['parameters']:
        if param['type'] in parameters:
            parameters[param['type']].append(param['name'])
        else:
            parameters[param['type']] = [param['name']]

    # Generate possible object combinations
    object_combinations = None
    if 'object' in parameters.keys():
        if len(scene_struct['objects']) < len(parameters['object']):
            if verbose:
                print("Scene has too little objects for this template")
            return output_dict
        object_combinations = generate_object_combinations(
            scene_struct, len(parameters['object']))

    # print(object_combinations)

    # First UNIQUE and NON_UNIQUE generation
    unq_descriptors, non_unq_descriptors = Constraints.unique(
        scene_struct, metadata['properties'], extra_unique_steps,
        return_uniques=("UNIQUE" in constraints.keys()),
        return_non_uniques=("NON_UNIQUE" in constraints.keys())
    )

    # print(unq_descriptors)
    # print(non_unq_descriptors)

    # UNIQUE, NON_UNIQUE filtering
    # Filter object combinations based on constraints
    # and uniqueness possibilities
    if "UNIQUE" in constraints.keys():
        object_combinations = Constraints.resolve_uniqueness(
            object_combinations,
            constraints['UNIQUE'],
            parameters['object'] if 'object' in parameters else None,
            unq_descriptors
        )
        if not validate_combinations(object_combinations, verbose):
            return output_dict
    # print(object_combinations)

    if "NON_UNIQUE" in constraints.keys():
        object_combinations = Constraints.resolve_uniqueness(
            object_combinations,
            constraints['NON_UNIQUE'],
            parameters['object'] if 'object' in parameters else None,
            non_unq_descriptors
        )
        if not validate_combinations(object_combinations, verbose):
            return output_dict
    # print(object_combinations)

    all_unq = []
    is_targets_unq = []
    is_targets_nunq = []
    if 'object' in parameters.keys():
        if 'UNIQUE' in constraints.keys():
            all_unq = [targ for sublist in constraints['UNIQUE'] for targ in sublist]
        if 'IS' in constraints.keys():
            is_targets_unq = [targ for targ in constraints['IS'] if targ[0]
                              in all_unq]
            is_targets_nunq = [targ for targ in constraints['IS'] if targ[0]
                               not in all_unq]

    # Resolve IS
    if len(is_targets_unq) > 0:
        object_combinations = Constraints.resolve_is(
            object_combinations,
            scene_struct,
            is_targets_unq,
            parameters['object'] if 'object' in parameters else None
        )
        if not validate_combinations(object_combinations, verbose):
            return output_dict

    # print(object_combinations)

    # Resolve IS_NOT
    if 'IS_NOT' in constraints.keys():
        object_combinations = Constraints.resolve_is_not(
            object_combinations,
            scene_struct,
            constraints['IS_NOT'],
            parameters['object'] if 'object' in parameters else None,
        )

    # print(object_combinations)

    # Resolve IN
    if 'IN' in constraints.keys():
        object_combinations = Constraints.resolve_in(
            object_combinations,
            scene_struct,
            constraints['IN'],
            parameters['object'] if 'object' in parameters else None,
        )

    # print(object_combinations)
    # Multiply non_unique descriptors for each object <> token
    # Cause IS for NON_UNIQUES may resolve different descriptors
    # for different tokens
    # e.g. <OBJ1> is not unique and is yellow, whereas
    # <OBJ2> is not unique and metal thermos
    # then <OBJ1> can be described with all non unique descriptors
    # which hold for all yellow objects
    # and <OBJ2> with all for metal thermoses

    nunq_tags = None
    if 'object' in parameters:
        if 'NON_UNIQUE' in constraints.keys():
            nunq_tags = [tag[0] for tag in constraints['NON_UNIQUE']]
            # print(constraints['NON_UNIQUE'])

    if nunq_tags is not None:
        non_unq_descriptors = (
            [copy.deepcopy(non_unq_descriptors)
             for i in range(len(nunq_tags))])

    # print(is_targets_nunq)

    # Resolve IS for set
    if len(is_targets_nunq) > 0:
        object_combinations, non_unq_descriptors = Constraints.resolve_is_set(
            object_combinations,
            scene_struct,
            metadata['properties'],
            is_targets_nunq,
            nunq_tags,
            non_unq_descriptors,
            parameters['object'] if 'object' in parameters else None
        )
        if not validate_combinations(object_combinations, verbose):
            return output_dict
    # print(object_combinations, non_unq_descriptors)

    # Resolve CONTAIN
    if nunq_tags is not None and len(nunq_tags) > 0:
        if 'CONTAIN' in constraints:
            non_unq_descriptors = Constraints.resolve_contain(
                scene_struct,
                non_unq_descriptors,
                nunq_tags,
                constraints['CONTAIN'],
                metadata['properties']
            )

    relation_descriptors = None
    # Relational clues
    if "RELATION_UNIQUE" in constraints.keys():
        object_combinations, relation_descriptors = Constraints.resolve_relation_unq(
            object_combinations,
            scene_struct,
            metadata['properties'],
            constraints['RELATION_UNIQUE'],
            parameters['object'] if 'object' in parameters else None,
            parameters['relation'] if 'relation' in parameters else None,
            non_unq_descriptors,
            nunq_tags
        )
        if not validate_combinations(object_combinations, verbose):
            return output_dict

    # print(relation_descriptors)

    table_parts_tags = parameters['table_part'] if 'table_part' in parameters else None
    table_parts = get_table_parts()
    table_parts_descriptors = None
    if table_parts_tags is not None:
        table_parts_descriptors = [copy.deepcopy(table_parts) for i in range(len(table_parts_tags))]
    # print(table_parts_descriptors)


    if "NOT_TP" in constraints.keys():
        table_parts_descriptors = Constraints.resolve_not_tp(
            object_combinations,
            scene_struct,
            metadata['properties'],
            constraints,
            parameters['object'] if 'object' in parameters else None,
            non_unq_descriptors,
            nunq_tags,
            table_parts_tags,
            table_parts_descriptors
        )

    # print(constraints)
    # print(table_parts_descriptors)

    if "TP_SET" in constraints.keys():
        table_parts_descriptors = Constraints.resolve_tp_set(
            scene_struct,
            constraints['TP_SET'],
            table_parts_tags,
            table_parts_descriptors
        )

    # print(table_parts_descriptors)

    if "NOT_TP_SET" in constraints.keys():
        table_parts_descriptors = Constraints.resolve_not_tp_set(
            scene_struct,
            constraints['NOT_TP_SET'],
            table_parts_tags,
            table_parts_descriptors
        )

    if "TP_NOT_EMPTY" in constraints.keys():
        table_parts_descriptors = Constraints.resolve_tp_not_empty(
            scene_struct,
            constraints['TP_NOT_EMPTY'],
            table_parts_tags,
            table_parts_descriptors
        )

    # print(table_parts_descriptors)

    # print(object_combinations)
    # print(relation_descriptors)
    # print(table_parts_descriptors)
    # print(non_unq_descriptors)

    # Final rejections
    # based on: object_combinations, unq_descriptors, non_unq_descriptors,
    # relation_descriptors, table_part_descriptors
    # to remove obvious not working ones

    combinations_dict = Constraints.descriptor_rejections(
        object_combinations,
        unq_descriptors,
        non_unq_descriptors,
        relation_descriptors,
        table_parts_descriptors,
        parameters,
        constraints,
        nunq_tags,
        table_parts_tags
    )

    # print(parameters)
    if 'weight_specifier' in parameters:
        if combinations_dict is not None:
            for v in combinations_dict.values():
                for ws in parameters['weight_specifier']:
                    v[ws] = get_weight_specifiers()

    if 'table_part' in parameters:
        if combinations_dict is not None:
            for v in combinations_dict.values():
                # print(v)
                for tp in parameters['table_part']:
                    if tp not in v:
                        v[tp] = get_table_parts()
    # print(combinations_dict)
    # print(combinations_dict[list(combinations_dict.keys())[0]])

    if combinations_dict is None:
        if verbose:
            print("Couldn't find any possible combination of parameters")
        return output_dict

    reject_combinations = len(combinations_dict.keys()) >= num_instructions

    for i in range(num_instructions):
        substitution_dict, combinations_dict, prog = get_token_substitution(
            combinations_dict,
            parameters,
            scene_struct,
            metadata['properties'],
            reject_combinations,
            # template['task'],
            template['program'],
            constraints
        )
        if prog is None:
            continue
        # print(substitution_dict, prog, task)

        # Get random text:
        text = random.choice(template['text'])
        for tag, val in substitution_dict.items():
            text = text.replace(tag, val)
        text = text.replace('_', ' ')
        text = replace_optionals(text)
        text = adjust_substitutions(text, plurals)
        text = adjust_substitutions(text, substitutions)
        # print(text)

        outp = {}
        outp['text'] = text
        outp['programs'] = prog
        # outp['task'] = task
        # print(prog[-1]['output'])
        if len(prog[-1]['output']) > 0:
            output_dict.append(outp)
        # print(prog[-1])
        # exit()
    # Generate programs / indicators of object in question
    # Generate high level task, e.g. pick::0, pick:[0, 1, 2]

    # Substitute for text
    # Make all text thingies

    return output_dict


def adjust_substitutions(text, substitutions):
    for plural_sub in substitutions.keys():
        if plural_sub in text:
            # print(plural_sub)
            text = text.replace(plural_sub, substitutions[plural_sub])
    return text


def replace_optionals(text):
    pattern = re.compile(r'\[([^\[]*)\]')

    while True:
        match = re.search(pattern, text)
        if not match:
            break
        i0 = match.start()
        i1 = match.end()
        if random.random() > 0.5:
            text = text[:i0] + match.groups()[0] + text[i1:]
        else:
            text = text[:i0] + text[i1:]

    text = text.replace('  ', ' ')
    return text


def get_token_substitution(combinations_dict, parameters,
                           scene_struct, property_names,
                           reject_combinations,
                           progs, constraints):
    
    # print(combinations_dict)
    # print(parameters)
    # Get reversed param dict
    # task_details = copy.deepcopy(task)
    # programs_details = copy.deepcopy(programs)
    param_types = {}
    for p_type, param_list in parameters.items():
        for t in param_list:
            param_types[t] = p_type
    # print(param_types)
    # Firstly get token set randomly from dict
    # print(combinations_dict)
    rand_pos = random.randint(0, len(combinations_dict.keys()) - 1)
    rand_key = list(combinations_dict.keys())[rand_pos]
    # print(combinations_dict.keys())
    # print(rand_key)
    # print(parameters)
    # task_tags = []
    # for subtask in task:
    #     for targs in subtask.values():
    #         if isinstance(targs, list):
    #             for targ in targs:
    #                 if targ in param_types.keys():
    #                     task_tags.append(targ)
    #         else:
    #             if targs in param_types.keys():
    #                 task_tags.append(targs)
    # prog_tags = programs.keys()
    # print(task)
    # print(task_tags)

    def get_obj_description(idx, descriptor):
        target_obj = scene_struct['objects'][idx]
        descr_list_adj = []
        descr_list_noun = []
        for prop_idx in descriptor:
            prop_name = property_names[prop_idx]
            if prop_name in ['name']:
                descr_list_noun.append(target_obj[prop_name])
            else:
                descr_list_adj.append(target_obj[prop_name])
        # print(rand_key, descriptor, descr_list_adj, descr_list_noun)
        descr_str_adj = ', '.join(descr_list_adj)
        if len(descr_list_noun) > 0:
            descr_str_noun = ' ' + ' '.join(descr_list_noun)
        else:
            descr_str_noun = ' ' + 'object'
        descr_str = descr_str_adj + descr_str_noun
        descr_str = descr_str.strip()
        return descr_str

    descriptors_dict = combinations_dict[rand_key]
    subs_dict = {}
    descr_dict = {}
    for tags, values in descriptors_dict.items():
        # print(tags, values)
        if isinstance(tags, tuple):
            # print(tags, values)
            rand_pos = random.randint(0, len(values[0]) - 1)
            # print(tags)
            # print(rand_pos)
            for i, tag in enumerate(tags):
                val = values[i][rand_pos]
                if param_types[tag] == 'object':
                    descr_dict[tag] = (rand_key[parameters['object'].index(tag)], val)
                    val = get_obj_description(rand_key[parameters['object'].index(tag)], val)
                else:
                    val = random.choice(val)
                    descr_dict[tag] = val
                subs_dict[tag] = val
                # if tag in task_tags:

        else:
            rand_pos = random.randint(0, len(values) - 1)
            val = values[rand_pos]
            # print(param_types[tags])
            descr_dict[tags] = val
            if param_types[tags] == 'object':
                descr_dict[tags] = (rand_key[parameters['object'].index(tags)], val)
                val = get_obj_description(rand_key[parameters['object'].index(tags)], val)
            subs_dict[tags] = val

    if reject_combinations:
        combinations_dict.pop(rand_key)

    # print('asd', descr_dict)
    # print(subs_dict)

    # print(subs_dict, descr_dict)
    programs_details = programs.generate_programs(
        progs, scene_struct, descr_dict, property_names)

    # task_details = programs.generate_task(
    #     task, programs_details, subs_dict)
    # print('a', combinations_dict)

    return subs_dict, combinations_dict, programs_details


def generate_object_combinations(scene_struct, num_objects):
    # Return all possible tuples of lenght num_objects
    # out of indices up to number of objects in the scene
    object_combinations = []
    if num_objects > 0:
        # print(len(scene_struct['objects']))
        # print(num_objects)
        object_combinations = list(itertools.permutations(
            range(len(scene_struct['objects'])), num_objects))
        # print(object_combinations)
    return object_combinations


def validate_combinations(object_combinations, verbose):
    if object_combinations is None:
        print("Some templates require correcting, exiting")
        exit()
    if len(object_combinations) < 1:
        if verbose:
            print("No possible object combinations fulfilling constraints")
        return False
    return True


def get_table_parts():
    # return ['left', 'right', 'front', 'back']
    return ['left', 'right']

def get_weight_specifiers():
    return ['lightest', 'heaviest']

def adjust_plurals(text, plurals):
    for word in text.split():
        word_adj = ''.join(x for x in word if x.isalpha())
        if word_adj in plurals:
            text = text.replace(word_adj, plurals[word_adj])
    return text


def load_scenes(config):
    with open(config.input_scene_file, 'r') as f:
        scene_data = json.load(f)
        scenes = scene_data["scenes"]
    first_idx = config.start_idx
    if config.num_scenes > 0:
        end_idx = first_idx + config.num_scenes
        scenes = scenes[first_idx:end_idx]
    else:
        scenes = scenes[first_idx:]
    return scenes, scene_data['info']


def load_templates(config):
    num_templates = 0
    templates = {}
    templ_individual_idx = {}
    i = 0
    if config.template_file is not None:
        with open(config.template_file, 'r') as f:
            for idx, template in enumerate(json.load(f)):
                num_templates += 1
                key = (template["template_id"], idx)
                templates[key] = template
                templ_individual_idx[key] = i
                i += 1
    else:
        for fname in sorted(os.listdir(config.template_dir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(config.template_dir, fname), 'r') as f:
                for idx, template in enumerate(json.load(f)):
                    num_templates += 1
                    key = (template["template_id"], idx)
                    templates[key] = template
                    templ_individual_idx[key] = i
                    i += 1
    return templates, templ_individual_idx, num_templates


def validate_templates(templates, metadata):
    templ_check = True

    for templ_id, templ in templates.items():
        # Validate parameter types
        param_dict = {}
        for param in templ['parameters']:
            if param['type'] not in metadata['parameters']:
                print(("Template from type {}, number {} containing"
                       + " unrecognised parameter type '{}' for '{}'.").format(
                    templ_id[0], templ_id[1], param['type'], param['name']))
                templ_check = False
            if param['name'] not in param_dict:
                param_dict[param['name']] = param['type']
            else:
                print(("Parameter {} double defined in template from "
                       + "type {}, number {}").format(
                    param['name'], templ_id[0], templ_id[1]))
        # Validate constraints
        for constr in templ['constraints']:
            if constr['type'] not in metadata['constraint_types']:
                print(("Template from type {}, number {} containing"
                       + " unrecognised constraint type '{}'.").format(
                    templ_id[0], templ_id[1], constr['type']))
                templ_check = False
            if not check_constriant_inputs(constr, param_dict):
                print(("Incorrect input for constraint of type '{}' in "
                       + "template from type {}, number {}").format(
                    constr['type'], templ_id[0], templ_id[1]))
                templ_check = False
    return templ_check


def check_constriant_inputs(constraint, param_types):
    correct = True
    constr_type = constraint['type']
    constr_target = constraint['target']

    if constr_type in ["UNIQUE", "NON_UNIQUE"]:
        if len(constr_target) != 1:
            print("Incorrect number of inputs")
            correct = False
        if (constr_target[0] not in param_types
                or param_types[constr_target[0]] != "object"):
            print("Incorrect type of the target")
            correct = False
    elif constr_type == "SCENE_UNIQUE":
        if len(constr_target) != 1:
            print("Incorrect number of inputs")
            correct = False
        # Check if string is more or less correct for now
        targ_split = constr_target[0].split('::')
        if len(targ_split) > 2:
            print("Incorrect type of the target, position {}".format(1))
            correct = False
    elif constr_type == "SCENE_SHARE_PROP":
        if len(constr_target) != 2:
            print("Incorrect number of inputs")
            correct = False
        # Check if string is more or less correct for now
        targ_split = constr_target[0].split('::')
        if len(targ_split) > 2:
            print("Incorrect type of the target, position {}".format(1))
            correct = False
        targ_split = constr_target[1].split('::')
        if len(targ_split) > 2:
            print("Incorrect type of the target, position {}".format(1))
            correct = False
    elif constr_type == "RELATION_UNIQUE":
        if len(constr_target) != 3:
            print("Incorrect number of inputs")
            correct = False
        if (constr_target[0] not in param_types
                or param_types[constr_target[0]] != "object"):
            print("Incorrect type of the target, position {}".format(0))
            correct = False
        if (constr_target[1] not in param_types
                or param_types[constr_target[1]] != "relation"):
            print("Incorrect type of the target, position {}".format(1))
            correct = False
        if (constr_target[2] not in param_types
                or param_types[constr_target[2]] != "object"):
            print("Incorrect type of the target, position {}".format(2))
            correct = False
    elif constr_type in ["IS", "IS_NOT", "CONTAIN"]:
        if len(constr_target) != 2:
            print("Incorrect number of inputs")
            correct = False
        if (constr_target[0] not in param_types
                or param_types[constr_target[0]] != "object"):
            print("Incorrect type of the target, position {}".format(0))
            correct = False
        # Check if string is more or less correct for now
        targ_split = constr_target[1].split('::')
        if len(targ_split) > 2:
            print("Incorrect type of the target, position {}".format(1))
            correct = False
    elif constr_type == "NOT_TP":
        if len(constr_target) != 2:
            print("Incorrect number of inputs")
            correct = False
        if (constr_target[0] not in param_types
                or param_types[constr_target[0]] != "object"):
            print("Incorrect type of the target, position {}".format(0))
            correct = False
        if (constr_target[1] not in param_types
                or param_types[constr_target[1]] != "table_part"):
            print("Incorrect type of the target, position {}".format(0))
            correct = False
    elif constr_type == "TP_NOT_EMPTY":
        if len(constr_target) != 1:
            print("Incorrect number of inputs")
            correct = False
        if (constr_target[0] not in param_types
                or param_types[constr_target[0]] != "table_part"):
            print("Incorrect type of the target, position {}".format(0))
            correct = False
    elif constr_type in ["TP_SET", "NOT_TP_SET"]:
        if len(constr_target) != 2:
            print("Incorrect number of inputs")
            correct = False
        if (constr_target[0] not in param_types
                or param_types[constr_target[0]] != "table_part"):
            print("Incorrect type of the target, position {}".format(0))
            correct = False
        targ_split = constr_target[1].split('::')
        if len(targ_split) > 2:
            print("Incorrect type of the target, position {}".format(1))
            correct = False
    elif constr_type == "IN":
        if len(constr_target) != 3:
            print("Incorrect number of inputs")
            correct = False
        if (constr_target[0] not in param_types
                or param_types[constr_target[0]] != "object"):
            print("Incorrect type of the target, position {}".format(0))
            correct = False
        if (constr_target[2] not in param_types
                or param_types[constr_target[2]] != "object"):
            print("Incorrect type of the target, position {}".format(0))
            correct = False
    return correct


def main(args):
    unknown_args = args[1]
    args = args[0]
    config = Config()
    config.merge_from_json(args.config)
    config.merge_from_cmd(unknown_args)
    print(config)
    with open(config.metadata, 'r') as f:
        metadata = json.load(f)
    with open(config.synonyms, 'r') as f:
        synonyms = json.load(f)
    with open(config.plurals, 'r') as f:
        plurals = json.load(f)
    with open(config.substitutions, 'r') as f:
        substitutions = json.load(f)

    os.makedirs(
        os.path.split(config.output_instruction_file)[0], exist_ok=True
    )

    # Load templates
    templates, templ_individual_idx, num_templates = load_templates(config)
    print("Read {} templates from directory {}".format(num_templates,
                                                       config.template_dir))

    if not validate_templates(templates, metadata):
        print("Please resolve template errors")
        return

    def reset_statistics():
        template_counts = {}
        tasks_counts = {}
        for key, template in templates.items():
            template_counts[template["template_id"]] = 0
            tasks_counts[template["task_id"]] = 0
        return template_counts, tasks_counts

    template_counts, tasks_counts = reset_statistics()
    # print(template_counts)
    # print(tasks_counts)

    scenes, scenes_info = load_scenes(config)
    instructions = []
    scene_counter = 0
    instructions_counter = 0
    num_scenes = len(scenes)
    print_digits = len(str(num_scenes))
    print_digits = ":0" + str(print_digits) + "d"
    message = "Starting image {}\t"
    message += "[{" + print_digits + "} / {" + print_digits + "}]"

    if config.templates_per_scene > 0:
        templates_per_scene = config.templates_per_scene
    else:
        templates_per_scene = len(templates)

    file_list = []

    for i, scene in enumerate(scenes):
        scene_cp = copy.deepcopy(scene)
        if scene_counter % config.reset_statistics == 0:
            print("Resetting distribution")
            reset_statistics()

        print(message.format(
            scene_cp["image_filename"], i + 1, len(scenes)
        ))

        scene_counter += 1
        template_list = list(templates.items())
        random.shuffle(template_list)
        if config.task_distribution:
            template_list = sorted(
                template_list,
                key=lambda x: template_counts[x[0][0]]
            )
            template_list = sorted(
                template_list,
                key=lambda x: tasks_counts[x[1]['task_id']]
            )
        else:
            template_list = sorted(
                template_list,
                key=lambda x: tasks_counts[x[1]['task_id']]
            )
            template_list = sorted(
                template_list,
                key=lambda x: template_counts[x[0][0]]
            )

        num_successful_templates = 0

        for (template_id, template_internal_idx), template in template_list:
            if config.verbose:
                print("Trying template\t id: {}\tidx: {}\ttask: {}".format(
                    template_id, template_internal_idx, template["task_id"]
                ))
            if config.timer and config.verbose:
                tic = time.time()
            output_dict = generate_instruction_from_template(
                scene_cp,
                template,
                metadata,
                synonyms,
                plurals,
                substitutions,
                num_instructions=config.instructions_per_template,
                extra_unique_steps=config.extra_steps,
                verbose=config.verbose
            )
            if config.timer and config.verbose:
                print("Time elapsed: {}".format(time.time() - tic))
            image_index = os.path.splitext(scene_cp["image_filename"])[0]
            image_index = int(image_index.split('_')[-1])
            if "split" in scenes_info:
                split = scenes_info['split']
            else:
                split = "new"
            templ_key = (template_id, template_internal_idx)
            for outp in output_dict:
                instructions.append({
                    'split': split,
                    'image_filename': scene_cp["image_filename"],
                    'image_index': image_index,
                    'instruction': outp['text'],
                    'program': outp['programs'],
                    # 'task': outp['task'],
                    'template_id': template_id,
                    'task_id': template["task_id"],
                    'template_internal_idx': template_internal_idx,
                    'template_overall_idx': templ_individual_idx[templ_key],
                    'instruction_idx': len(instructions)
                })
            if len(output_dict) > 0:
                if config.verbose:
                    print("Generated {} instructions".format(len(output_dict)))
                num_successful_templates += 1
                template_counts[templ_key[0]] += 1
                tasks_counts[template['task_id']] += len(output_dict)
            elif config.verbose:
                print("No instructions generated from template")
            if num_successful_templates >= templates_per_scene:
                break
        # Delete that copy
        del(scene_cp)
        # Intermediate saving
        if config.dump_freq > 0:
            if (i + 1) % config.dump_freq == 0:
                base, ext = os.path.splitext(config.output_instruction_file)
                num_range = "_{:06d}_{:06d}".format(
                    i + 1 - config.dump_freq, i)
                filename = base + num_range + ext
                file_list.append(filename)
                print("Intermediate save at: {}".format(filename))
                with open(filename, 'w') as f:
                    json.dump({
                        'info': scenes_info,
                        'instructions': instructions
                    }, f)
                instructions_counter += len(instructions)
                instructions = []
        else:
            instructions_counter = len(instructions)

    print("Finished! Generated {} instructions".format(instructions_counter))
    if config.dump_freq > 0:
        collect_instructions(
            file_list,
            config.output_instruction_file,
            scenes_info
        )
    else:
        with open(config.output_instruction_file, 'w') as f:
            print("Writing output to {}".format(
                config.output_instruction_file))
            json.dump({
                'info': scenes_info,
                'instructions': instructions
            }, f, indent=4)


if __name__ == '__main__':
    known_args, unknown_args = parser.parse_known_args()
    main((known_args, unknown_args))
