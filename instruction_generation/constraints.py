import itertools
import collections


class Constraints:
    def unique(scene_struct, property_names, extra_steps,
               return_uniques=True, return_non_uniques=True):
        # Generate unique descriptors for all objects
        # Get unique descriptors of the shortest
        # and up to shortest + extra_steps length
        # (to increase randomness a bit)
        # Return idxs of unique properties as in property_names

        if not return_uniques and not return_non_uniques:
            return None, None

        # First, create matrix of properties
        prop_matrix = [
            [o[p_name] for p_name in property_names]
            for o in scene_struct['objects']
        ]
        # print(prop_matrix)

        if return_uniques:
            descriptor_idxs_unq = [[]
                                   for i in range(len(prop_matrix))]
        else:
            descriptor_idxs_unq = None
        if return_non_uniques:
            descriptor_idxs_nunq = [[]
                                    for i in range(len(prop_matrix))]
        else:
            descriptor_idxs_nunq = None

        steps_unique = [0] * len(scene_struct['objects'])
        steps_non_unique = [0] * len(scene_struct['objects'])

        # Go through all possible lenghts of descriptors
        for i in range(len(property_names)):
            num_feats = i + 1

            if return_uniques:
                unique_indicator = [False] * len(scene_struct['objects'])
            if return_non_uniques:
                non_unique_indicator = [False] * len(scene_struct['objects'])

            # Generate all combinations
            for idx_set in itertools.combinations(range(len(property_names)),
                                                  num_feats):

                # Pick chosen combination
                property_reduced = [
                    tuple([obj_prop[idx] for idx in idx_set])
                    for obj_prop in prop_matrix
                ]
                cnt = collections.Counter(property_reduced)

                # Count uniques (which objs are uniquely described)
                unique_idxs = [property_reduced.index(prop)
                               for prop in property_reduced if cnt[prop] == 1]
                if return_non_uniques:
                    non_unique_idxs = [idx
                                       for idx in range(len(prop_matrix))
                                       if idx not in unique_idxs]

                # Update unique indicator for that length
                if return_uniques:
                    unique_indicator = [
                        unique_indicator[i] or (True
                                                if i in unique_idxs
                                                else False)
                        for i in range(len(unique_indicator))]
                    for idx in unique_idxs:
                        if steps_unique[idx] < extra_steps + 1:
                            descriptor_idxs_unq[idx].append(idx_set)
                if return_non_uniques:
                    non_unique_indicator = [
                        non_unique_indicator[i] or (True
                                                    if i in non_unique_idxs
                                                    else False)
                        for i in range(len(non_unique_indicator))]
                    for idx in non_unique_idxs:
                        if steps_non_unique[idx] < extra_steps + 1:
                            descriptor_idxs_nunq[idx].append(idx_set)
            if return_uniques:
                steps_unique = [steps_unique[i] + 1
                                if unique_indicator[i]
                                else steps_unique[i]
                                for i in range(len(steps_unique))]
            if return_non_uniques:
                steps_non_unique = [steps_non_unique[i] + 1
                                    if non_unique_indicator[i]
                                    else steps_non_unique[i]
                                    for i in range(len(steps_non_unique))]

        # print(descriptor_idxs_unq)
        # print(descriptor_idxs_nunq)
        return descriptor_idxs_unq, descriptor_idxs_nunq

    def resolve_uniqueness(object_combinations, unq_targets,
                           object_tags, uniqueness_descriptor):
        if object_tags is None:
            print("No tag for the constraint to be applied, correct template")
            return None
        # Check target correctness
        if not all([len(targ) > 0 and targ[0] in object_tags
                    for targ in unq_targets]):
            print("Incorrect targets for UNIQUE constraint, correct template")
            return None
        # Get idxs of objects in combinations for which uniqueness
        # has to be applied
        tag_idxs = [object_tags.index(targ[0]) for targ in unq_targets]
        # Go through combinations and filter out not applicable ones
        filtered_combinations = []
        for comb in object_combinations:
            keep = True
            for idx in tag_idxs:
                if len(uniqueness_descriptor[comb[idx]]) < 1:
                    keep = False
            if keep:
                filtered_combinations.append(comb)
        return filtered_combinations

    def resolve_is(object_combinations, scene_struct,
                   is_targets, object_tags):
        if object_tags is None:
            print("No tag for the constraint to be applied, correct template")
            return None
        # Check target correctness
        if not all([len(targ) == 2 and targ[0] in object_tags
                    for targ in is_targets]):
            print("Incorrect targets for IS constraint, correct template")
            return None
        # Get idxs of objects in combinations for which IS
        # has to be applied
        tag_idxs = [object_tags.index(targ[0]) for targ in is_targets]
        # print(tag_idxs)
        # Create bool matrix for all looked properties
        constraint_lookup = []
        for constr in is_targets:
            constr_body = constr[1]
            # print(constr_body)
            if '::' in constr_body:
                split = constr_body.split('::')
                query = split[0]
                targ = split[1]
            else:
                query = constr_body
                targ = True
            constraint_lookup.append([])
            for obj in scene_struct['objects']:
                if query not in obj:
                    # print(query)
                    print("Missing annotation in the scene")
                    return None
                constraint_lookup[-1].append(obj[query] == targ)
            # print(constraint_lookup)

        # Filter out impossible combinations
        filtered_combinations = []
        for comb in object_combinations:
            keep = True
            for i, idx in enumerate(tag_idxs):
                if not constraint_lookup[i][comb[idx]]:
                    keep = False
            if keep:
                filtered_combinations.append(comb)
        return filtered_combinations

    def resolve_is_set(object_combinations, scene_struct, property_names,
                       is_targets, object_tags, descriptors, all_objects):
        if object_tags is None:
            print("No tag for the constraint to be applied, correct template")
            return None
        # Check target correctness
        if not all([len(targ) == 2 and targ[0] in object_tags
                    for targ in is_targets]):
            print("Incorrect targets for IS constraint, correct template")
            return None
        # Get idxs of objects in combinations for which IS
        # has to be applied
        tag_idxs = [all_objects.index(targ[0]) for targ in is_targets]
        tag_idxs_descr = [object_tags.index(targ[0]) for targ in is_targets]
        # print(tag_idxs)
        # print(tag_idxs_descr)
        # Create bool matrix for all looked properties
        constraint_lookup = []
        for constr in is_targets:
            constr_body = constr[1]
            # print(constr_body)
            if '::' in constr_body:
                split = constr_body.split('::')
                query = split[0]
                targ = split[1]
            else:
                query = constr_body
                targ = True
            constraint_lookup.append([])
            for obj in scene_struct['objects']:
                if query not in obj:
                    # print(query)
                    print("Missing annotation in the scene")
                    return None
                constraint_lookup[-1].append(obj[query] == targ)

        # print(constraint_lookup)
        # print(object_combinations)

        # Filter out impossible combinations
        filtered_combinations = []
        for comb in object_combinations:
            keep = True
            for i, idx in enumerate(tag_idxs):
                if not constraint_lookup[i][comb[idx]]:
                    keep = False
            if keep:
                filtered_combinations.append(comb)

        # print(filtered_combinations)
        # Filter out descriptors
        constr_dict = {}
        for constr in is_targets:
            constr_obj = constr[0]
            constr_body = constr[1]
            if '::' in constr_body:
                split = constr_body.split('::')
                query = split[0]
                targ = split[1]
            else:
                query = constr_body
                targ = True
            if constr_obj not in constr_dict:
                constr_dict[constr_obj] = []
            constr_dict[constr_obj].append((query, targ))

        filtered_descriptors = descriptors
        # print(filtered_descriptors)
        # print(constr_dict)
        # print(object_tags)
        for tag_idx_descr, tag_idx, tag in zip(tag_idxs_descr, tag_idxs, object_tags):
            if tag in constr_dict:
                tag_constr = constr_dict[tag]
                tag_descriptors = filtered_descriptors[tag_idx_descr]
                for obj_idx, descr_set in enumerate(tag_descriptors):
                    # Check if object can appear under that tag,
                    # if not, don't bother cause it will not be used anyway
                    if not any([comb[tag_idx] == obj_idx
                                for comb in filtered_combinations]):
                        continue
                    # Check each description
                    for descr in reversed(descr_set):
                        objects = scene_struct['objects']
                        # Get objects that match description
                        for feat in descr:
                            p_name = property_names[feat]
                            prop = (scene_struct['objects'][obj_idx]
                                    [property_names[feat]])
                            objects = ([obj for obj in objects
                                        if obj[p_name] == prop])
                        obj_check = [True for obj in objects]
                        for constr in tag_constr:
                            obj_check = [
                                obj_check[i] and (obj[constr[0]] == constr[1])
                                for i, obj in enumerate(objects)
                            ]
                        if not all(obj_check):
                            descr_set.remove(descr)

        # print(filtered_descriptors)
        return filtered_combinations, filtered_descriptors

    def resolve_relation_unq(object_combinations, scene_struct, property_names,
                             relation_targets, object_tags, relation_tags,
                             descriptors, descr_tags):
        # Take combinations
        # Find unique relations
        # Return combinations and matching possible
        if object_tags is None:
            print("No tag for the constraint to be applied, correct template")
            return None

        # Go through combinations and filter out not applicable ones
        filtered_combinations = object_combinations
        relations = []

        for c, constr in enumerate(relation_targets):
            relations.append([])
            relations[c] = [[] for i in range(len(object_combinations))]
            for i, comb in reversed(list(enumerate(object_combinations))):
                # print(i, comb)
                source_obj_idx = comb[object_tags.index(constr[2])]
                related_obj_idx = comb[object_tags.index(constr[0])]
                # print(source_obj_idx, related_obj_idx)
                related_obj_descriptors = descriptors[descr_tags.index(constr[0])][related_obj_idx]
                relations[c][i] = [[] for i in range(len(related_obj_descriptors))]
                source_obj = scene_struct['objects'][source_obj_idx]
                for relation in ['left', 'right', 'front', 'behind']:
                    # print(source_obj['directions'])
                    if related_obj_idx not in source_obj['directions'][relation]:
                        continue
                    for k, descr in enumerate(related_obj_descriptors):
                        descr_check = []
                        # print(descr, scene_struct['objects'][related_obj_idx]['name'])
                        desired_props = []
                        for feat in descr:
                            prop = scene_struct['objects'][related_obj_idx][property_names[feat]]
                            desired_props.append(prop)
                        for o in source_obj['directions'][relation]:
                            # print(source_obj['name'], relation, scene_struct['objects'][o]['name'])
                            descr_check.append([])
                            for feat in descr:
                                descr_check[-1].append(scene_struct['objects'][o][property_names[feat]])
                        # print(set((tuple(lst) for lst in descr_check)))
                        desired_props = tuple(desired_props)
                        descr_check = [tuple(lst) for lst in descr_check]
                        # print(desired_props, descr_check)
                        descr_check = [tup == desired_props for tup in descr_check]
                        # print(desired_props, sum(descr_check))
                        # print(set((tuple(lst) for lst in descr_check)))
                        # if len(set((tuple(lst) for lst in descr_check))) == 1:
                            # relations[c][i][k].append(relation)
                        if sum(descr_check) == 1:
                            relations[c][i][k].append(relation)

        # print(filtered_combinations)
        # print(relations)

        return filtered_combinations, relations

    def resolve_not_tp(object_combinations, scene_struct, property_names, constraints,
                       object_tags, descriptors, descr_tags, table_tags, table_descr):

        not_tp_constrs = constraints['NOT_TP']
        if 'UNIQUE' in constraints.keys():
            uniques = [item[0] for item in constraints['UNIQUE']]
        else:
            uniques = []
        if 'NON_UNIQUE' in constraints.keys():
            non_uniques = [item[0] for item in constraints['NON_UNIQUE']]
        else:
            non_uniques = []

        table_descr = [[] for i in range(len(table_tags))]
        tags_included = []
        for targ in not_tp_constrs:
            if targ[0] in tags_included:
                continue
            table_idx = table_tags.index(targ[1])
            t_descrs = table_descr[table_idx]
            if targ[0] in uniques:
                t_descrs = [list(get_table_parts()) for c in range(len(object_combinations))]
            elif targ[0] in non_uniques:
                comb_idx = object_tags.index(targ[0])
                for comb in object_combinations:
                    obj_idx = comb[comb_idx]
                    d_idx = descr_tags.index(targ[0])
                    descrs = descriptors[d_idx][obj_idx]
                    # print(descrs)
                    t_descrs.append([list(get_table_parts()) for d in descrs])
            table_descr[table_idx] = t_descrs
            tags_included.append(targ[0])

        for targ in not_tp_constrs:
            # Check if UNIQUE or NOT
            table_idx = table_tags.index(targ[1])
            t_descrs = table_descr[table_idx]
            comb_idx = object_tags.index(targ[0])
            for c, comb in enumerate(object_combinations):
                obj_idx = comb[comb_idx]
                if targ[0] in uniques:
                    diff = get_table_parts() - set(scene_struct['objects'][obj_idx]['table_part'])
                    t_descrs[c] = list(set.intersection(set(diff), set(t_descrs[c])))
                elif targ[0] in non_uniques:
                    # Get descriptors
                    d_idx = descr_tags.index(targ[0])
                    descrs = descriptors[d_idx][obj_idx]
                    for d, desc in enumerate(descrs):
                        # For each descriptor check table parts NOT common for all items
                        # described by descriptor (aka common denominator of diffs)
                        # print(desc)
                        desired_props = [scene_struct['objects'][obj_idx][property_names[f]] for f in desc]
                        diffs_list = []
                        for i, obj in enumerate(scene_struct['objects']):
                            keep = True
                            for f, feat in enumerate(desc):
                                if obj[property_names[feat]] != desired_props[f]:
                                    keep = False
                            if keep:
                                diffs_list.append(get_table_parts() - set(obj['table_part']))
                        matching_tps = list(set.intersection(*map(set, diffs_list)))
                        # print(t_descrs)
                        t_descrs[c][d] = list(set.intersection(set(matching_tps), set(t_descrs[c][d])))
                table_descr[table_idx] = t_descrs

        return table_descr

    def resolve_tp_set(scene_struct, constraint,
                       table_parts_tags, table_parts_descriptors):
        # print(constraint)
        for constr in constraint:
            # print(constr)
            descr_idx = table_parts_tags.index(constr[0])
            descr = table_parts_descriptors[descr_idx]
            c_targ = constr[1]
            if '::' in c_targ:
                split = c_targ.split('::')
                query = split[0]
                targ = split[1]
            else:
                query = c_targ
                targ = True
            # print(query, targ)

            for tp in reversed(descr):
                for obj in scene_struct['objects']:
                    if tp in obj['table_part']:
                        if obj[query] != targ:
                            descr.remove(tp)
                            break

            table_parts_descriptors[descr_idx] = descr

        return table_parts_descriptors

    def resolve_tp_not_empty(scene_struct, constraint,
                             table_parts_tags, table_parts_descriptors):
        # print(constraint)
        for constr in constraint:
            # print(constr)
            descr_idx = table_parts_tags.index(constr[0])
            descr = table_parts_descriptors[descr_idx]

            for tp in reversed(descr):
                tp_present = False
                for obj in scene_struct['objects']:
                    if tp in obj['table_part']:
                        tp_present = True
                        break
                if not tp_present:
                    descr.remove(tp)

            table_parts_descriptors[descr_idx] = descr

        return table_parts_descriptors

    def resolve_not_tp_set(scene_struct, constraint,
                           table_parts_tags, table_parts_descriptors):
        # print(constraint)
        for constr in constraint:
            descr_idx = table_parts_tags.index(constr[0])
            descr = table_parts_descriptors[descr_idx]
            c_targ = constr[1]
            if '::' in c_targ:
                split = c_targ.split('::')
                query = split[0]
                targ = split[1]
            else:
                query = c_targ
                targ = True
            # print(query, targ)

            for tp in reversed(descr):
                for obj in scene_struct['objects']:
                    if tp in obj['table_part']:
                        if obj[query] == targ:
                            descr.remove(tp)
                            break

            table_parts_descriptors[descr_idx] = descr

        return table_parts_descriptors

    def resolve_contain(scene_struct, descriptors, non_unq_tags,
                        constraint, property_names):
        for constr in constraint:
            descr_idx = non_unq_tags.index(constr[0])
            c_targ = constr[1]
            if '::' in c_targ:
                split = c_targ.split('::')
                query = split[0]
                targ = split[1]
            else:
                query = c_targ
                targ = True
            descrs = descriptors[descr_idx]
            for i, obj_desc in enumerate(descrs):
                for desc in reversed(obj_desc):
                    desired_props = [scene_struct['objects'][i][property_names[f]] for f in desc]
                    keep = False
                    for obj in scene_struct['objects']:
                        corr_object = True
                        for f, feat in enumerate(desc):
                            if obj[property_names[feat]] != desired_props[f]:
                                corr_object = False
                        if corr_object:
                            if obj[query] == targ:
                                keep = True
                    if not keep:
                        obj_desc.remove(desc)
            descriptors[descr_idx] = descrs
        return descriptors

    def resolve_contain_tp(scene_struct, descriptors, non_unq_tags,
                        constraint, property_names):
        for constr in constraint:
            descr_idx = non_unq_tags.index(constr[0])
            c_targ = constr[1]
            if '::' in c_targ:
                split = c_targ.split('::')
                query = split[0]
                targ = split[1]
            else:
                query = c_targ
                targ = True
            descrs = descriptors[descr_idx]
            for i, obj_desc in enumerate(descrs):
                for desc in reversed(obj_desc):
                    desired_props = [scene_struct['objects'][i][property_names[f]] for f in desc]
                    keep = False
                    for obj in scene_struct['objects']:
                        corr_object = True
                        for f, feat in enumerate(desc):
                            if obj[property_names[feat]] != desired_props[f]:
                                corr_object = False
                        if corr_object:
                            if obj[query] == targ:
                                keep = True
                    if not keep:
                        obj_desc.remove(desc)
            descriptors[descr_idx] = descrs
        return descriptors

    def resolve_is_not(object_combinations, scene_struct, constraints, object_tags):
        # print(constraints)
        constraint_lookup = []
        constraint_targets = [c[0] for c in constraints]
        # print(constraint_targets)
        for constr in constraints:
            # constr_targ = constr[0]
            constr_val = constr[1]
            if '::' in constr_val:
                split = constr_val.split('::')
                query = split[0]
                targ = split[1]
            else:
                query = constr_val
                targ = True
            correctness_vec = []
            for obj in scene_struct['objects']:
                correctness_vec.append(obj[query] != targ)
            constraint_lookup.append(correctness_vec)
        # print(constraint_lookup)

        filtered_combinations = []
        for comb in object_combinations:
            keep = True
            for i, tag in enumerate(constraint_targets):
                if not constraint_lookup[i][comb[object_tags.index(tag)]]:
                    keep = False
            if keep:
                filtered_combinations.append(comb)

        return filtered_combinations

    def resolve_in(object_combinations, scene_struct, constraints, object_tags):
        # print(constraints)
        combinations_lookup = [True for i in range(len(object_combinations))]
        # print(combinations_lookup)
        for constr in constraints:
            query = constr[1]
            query_obj = constr[0]
            targ_obj = constr[2]
            for i, comb in enumerate(object_combinations):
                query_idx = comb[object_tags.index(query_obj)]
                targ_idx = comb[object_tags.index(targ_obj)]
                # print(comb, query_idx, targ_idx)
                if targ_idx not in scene_struct['objects'][query_idx][query]:
                    combinations_lookup[i] = False
        # print(combinations_lookup)
        filtered_combinations = [comb for comb, val in zip(object_combinations, combinations_lookup) if val]

        return filtered_combinations

    def check_scene_unique(constraints, scene_struct):
        correct = True
        for constr in constraints:
            constr_val = constr[0]
            if '::' in constr_val:
                split = constr_val.split('::')
                query = split[0]
                targ = split[1]
            else:
                query = constr_val
                targ = True
            correctness_vec = []
            for obj in scene_struct['objects']:
                if obj[query] == targ:
                    correctness_vec.append(True)
            if len(correctness_vec) != 1:
                correct = False

        return correct

    def check_scene_prop_share(constraints, scene_struct):
        correct = True
        for constr in constraints:
            constr_val1 = constr[0]
            if '::' in constr_val1:
                split = constr_val1.split('::')
                query1 = split[0]
                targ1 = split[1]
            else:
                query1 = constr_val1
                targ1 = True
            constr_val2 = constr[1]
            if '::' in constr_val2:
                split = constr_val2.split('::')
                query2 = split[0]
                targ2 = split[1]
            else:
                query2 = constr_val2
                targ2 = True
            correctness_vec = []
            for obj in scene_struct['objects']:
                if obj[query1] == targ1:
                    if obj[query2] == targ2:
                        correctness_vec.append(True)
                    else:
                        correctness_vec.append(False)
            if not all(correctness_vec):
                correct = False

        return correct

    def descriptor_rejections(object_combinations, unq_descriptors, non_unq_descriptors,
                              relation_descriptors, table_parts_descriptors,
                              parameters, constraints,
                              non_unq_tags, table_parts_tags):
        # print("\n\n\n")
        # print(object_combinations)
        # print(unq_descriptors)
        # print(non_unq_descriptors)
        # print(relation_descriptors)
        # print(table_parts_descriptors)
        # print(parameters)
        # print(constraints)
        # print(non_unq_tags)
        # print(table_parts_tags)

        if non_unq_tags is None:
            non_unq_tags = []
        if relation_descriptors is None:
            relation_descriptors = []

        # Divide table part to connected to objects, and not
        table_parts_obj_unq = []
        table_parts_obj_nunq = []
        table_parts_non_obj = []
        if 'table_part' in parameters.keys():
            for table_tag in parameters['table_part']:
                for constr_type, constr_targs in constraints.items():
                    if constr_type in ["NOT_TP"]:
                        # print(constr_targ)
                        for constr_targ in constr_targs:
                            if table_tag in constr_targ:
                                obj_tag = constr_targ[0]
                                if obj_tag in non_unq_tags:
                                    table_parts_obj_nunq.append(table_tag)
                                else:
                                    table_parts_obj_unq.append(table_tag)
                    elif constr_type in ["TP_SET", "NOT_TP_SET"]:
                        # print(constr_targ)
                        for constr_targ in constr_targs:
                            if table_tag in constr_targ:
                                table_parts_non_obj.append(table_tag)

        # print(table_parts_obj_unq)
        # print(table_parts_obj_nunq)
        # print(table_parts_non_obj)

        if 'object' in parameters.keys():
            combinations_lookup = []
            for c, comb in enumerate(object_combinations):
                keep = True
                for i, obj_idx in enumerate(comb):
                    tag = parameters['object'][i]
                    if tag not in non_unq_tags:
                        # Check unique descriptors
                        # (they shouldn't have changed, but maybe for future)
                        if len(unq_descriptors[obj_idx]) < 1:
                            keep = False
                    else:
                        # Check non unique descriptors
                        # (some were rejected by constraints so remove object
                        # combinations that have missing descriptors)

                        # Find tag index in non unique descriptors list
                        non_unq_tag_idx = non_unq_tags.index(tag)
                        # print(non_unq_descriptors)
                        if len(non_unq_descriptors[non_unq_tag_idx][obj_idx]) < 1:
                            keep = False
                # Check relation descriptors
                # (if 'relation' in parameters)
                # (for the combination there has to exist a possibility of
                # all relational tokens for it to work)
                for rel_tag_descr in relation_descriptors:
                    curr_comb_rel_descrs = rel_tag_descr[c]
                    # Find if there is a relation that would work for
                    # any of the non unique descriptors
                    # print('cd', any([len(cd) > 0 for cd in curr_comb_rel_descrs]))
                    if not any([len(cd) > 0 for cd in curr_comb_rel_descrs]):
                        keep = False
                # Check table part descriptors
                # (only those that have corresponding constraint)
                # check for uniques
                for tp_tag in table_parts_obj_unq:
                    tp_idx = parameters['table_part'].index(tp_tag)
                    tp_desc = table_parts_descriptors[tp_idx][c]
                    if len(tp_desc) < 1:
                        keep = False
                for tp_tag in table_parts_obj_nunq:
                    tp_idx = parameters['table_part'].index(tp_tag)
                    tp_desc = table_parts_descriptors[tp_idx][c]
                    if not any([len(tpd) > 0 for tpd in tp_desc]):
                        keep = False
                # print(keep)
                combinations_lookup.append(keep)

        combinations_dict = {}
        if 'object' in parameters.keys():
            for c, comb in enumerate(object_combinations):
                if not combinations_lookup[c]:
                    continue
                token_dict = {}
                keep = True
                for obj_tag in parameters['object']:
                    if obj_tag not in non_unq_tags:
                        obj_idx = comb[parameters['object'].index(obj_tag)]
                        token_dict[obj_tag] = unq_descriptors[obj_idx]
                    else:
                        obj_idx = comb[parameters['object'].index(obj_tag)]
                        obj_tokens = non_unq_descriptors[non_unq_tags.index(obj_tag)][obj_idx]
                        rel_tag = None
                        if 'RELATION_UNIQUE' in constraints.keys():
                            for tar in constraints['RELATION_UNIQUE']:
                                if obj_tag in tar:
                                    rel_tag = tar[1]
                        # print(obj_tag, rel_tag)
                        descr_correct = [True for i in range(len(obj_tokens))]
                        if rel_tag is not None:
                            rel_tag_idx = parameters['relation'].index(rel_tag)
                            rel_tokens = relation_descriptors[rel_tag_idx][c]
                            rel_tokens_filt = []
                            # print(obj_tokens, rel_tokens)
                            for t, tok in enumerate(rel_tokens):
                                if len(tok) < 1:
                                    descr_correct[t] = False
                        tp_tag = None
                        if 'NOT_TP' in constraints.keys():
                            for tar in constraints['NOT_TP']:
                                if obj_tag in tar:
                                    tp_tag = tar[1]
                        if tp_tag is not None:
                            tp_tag_idx = parameters['table_part'].index(tp_tag)
                            tp_tokens = table_parts_descriptors[tp_tag_idx][c]
                            # print(obj_tokens,tp_tokens)
                            tp_tokens_filt = []
                            # print(obj_tokens, rel_tokens)
                            for t, tok in enumerate(tp_tokens):
                                if len(tok) < 1:
                                    descr_correct[t] = False

                        obj_tokens_filt = []
                        for t, tok in enumerate(obj_tokens):
                            if descr_correct[t]:
                                obj_tokens_filt.append(tok)
                                if rel_tag is not None:
                                    rel_tokens_filt.append(rel_tokens[t])
                                if tp_tag is not None:
                                    tp_tokens_filt.append(tp_tokens[t])

                        if len(obj_tokens_filt) < 1:
                            keep = False
                        # print(obj_tag, rel_tag, tp_tag)
                        # print(obj_tokens, rel_tokens, tp_tokens)
                        # print(obj_tokens_filt, rel_tokens_filt, tp_tokens_filt)
                        if rel_tag is None and tp_tag is None:
                            token_dict[obj_tag] = obj_tokens_filt
                        elif rel_tag is not None and tp_tag is None:
                            token_dict[(obj_tag, rel_tag)] = (obj_tokens_filt, rel_tokens_filt)
                        elif rel_tag is None and tp_tag is not None:
                            token_dict[(obj_tag, tp_tag)] = (obj_tokens_filt, tp_tokens_filt)
                        else:
                            token_dict[(obj_tag, rel_tag, tp_tag)] = (
                                obj_tokens_filt, rel_tokens_filt, tp_tokens_filt)
                if len(table_parts_obj_unq) > 0:
                    for tp_tag in table_parts_obj_unq:
                        tp_tag_idx = parameters['table_part'].index(tp_tag)
                        tp_tokens = table_parts_descriptors[tp_tag_idx][c]
                        if len(tp_tokens) < 1:
                            keep = False
                        token_dict[tp_tag] = tp_tokens
                        # print(tp_tokens)
                if len(table_parts_non_obj) > 0:
                    for tp_tag in table_parts_non_obj:
                        tp_tag_idx = parameters['table_part'].index(tp_tag)
                        tp_tokens = table_parts_descriptors[tp_tag_idx]
                        if len(tp_tokens) < 1:
                            keep = False
                        token_dict[tp_tag] = tp_tokens
                        # print(tp_tokens)

                if not keep:
                    continue

                combinations_dict[comb] = token_dict
                # print(comb)
                # print(token_dict)
        else:
            keep = True
            token_dict = {}
            for tp_tag in parameters['table_part']:
                tp_tag_idx = parameters['table_part'].index(tp_tag)
                tp_tokens = table_parts_descriptors[tp_tag_idx]
                if len(tp_tokens) < 1:
                    keep = False
                token_dict[tp_tag] = tp_tokens
            if keep:
                combinations_dict[(-1, -1)] = token_dict
        # print(combinations_dict)
        if len(combinations_dict.keys()) < 1:
            combinations_dict = None
        return combinations_dict


def get_table_parts():
    # return set(['left', 'right', 'front', 'back'])
    return set(['left', 'right'])

