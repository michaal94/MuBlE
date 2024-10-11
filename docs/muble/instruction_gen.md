---
layout: default
title: Instruction generation
parent: CLIER
nav_order: 2
---

# Instruction generation
Instruction generation is performed by going to `instruction_generation` directory and calling:
```bash
cd instruction_generation
python generate_instructions.py -c config.json
```
where `config.json` is a path to configuration file.

We present a default/sample configuration file below:
```json
{
	"metadata": "configs/metadata.json",
	"template_dir": "templates/templates_neurips",
	"synonyms": "configs/synonyms.json",
	"plurals": "configs/plurals.json",
	"substitutions": "configs/substitutions.json",
	"input_scene_file": "../demo/demo_input/NS_AP_demo_scenes.json",
	"output_instruction_file": "../output/instruction_generation/NS_AP_demo_instructions.json",
	"start_idx": 0,
	"num_scenes": -1,
	"instructions_per_template": 1,
	"templates_per_scene": -1,
	"template_timeout": 60,
	"verbose": true,
	"dump_freq": -1,
	"reset_statistics": 250,
	"task_distribution": true,
	"timer": false,
	"extra_steps": 0
}
```

## Notable paratmeters
- `template_dir` or `template_file` -  a directory of a single file of templates can be provided
- `extra_steps` - a parameter related to choice of descriptor words (adjectives/nouns) for the given objects, increasing the parameter, increases the number of possible redundant descriptors (e.g. in the scene, where the only blue item is a mug, `extra_steps=0` results in `blue` being the only viable descriptor, `extra_steps=1` allows for `blue mug`)

## Templates
Instructions are generated based on templates. We show an example of template with explanation:
```json
{
    "text": [ 
        "Move the <OBJ1> to the <TP1> part of the table.",
        "Place the <OBJ1> on the <TP1> part of the table."
    ],                                  <-- Alternative forms of instruction phrasing
    "parameters": [
        {
            "name": "<OBJ1>",           <-- Relations between token and what they represent
            "type": "object"
        },
        {
            "name": "<TP1>",
            "type": "table_part"
        }
    ],
    "constraints": [                    <-- Constraints with their type and target
        {
            "target": [
                "<OBJ1>"
            ],
            "type": "UNIQUE"
        },
        {
            "target": [
                "<OBJ1>",
                "pickupable"
            ],
            "type": "IS"
        },
        {
            "target": [
                "<OBJ1>",
                "<TP1>"
            ],
            "type": "NOT_TP"
        }
    ],
    "program": [                        <-- Basic form of the symbolic program
        {
            "type": "scene",
            "input": null,
            "input_values": null
        },
        {
            "type": "filter_object",
            "input": -1,
            "input_values": "<OBJ1>"
        },
        {
            "type": "move_to",
            "input": -1,
            "input_values": "<TP1>"
        }
    ],
    "task_id": "move",                  <-- Use for grouping per task type
    "template_id": "move_single"        <-- Use for grouping per template type
}
```

## Available constraints
- **UNIQUE** (required) - main rejection mechanism, returns all possible shortest (or of given length) adjectives combinations that uniquely describe scene objects (\ie all descriptions that are applicable to the target token)
- **NON_UNIQUE** (required) - second main rejection mechanism, returns all possible shortest (or of given length) adjectives combinations that describe more than one object in the scene
- **IS** - used to set the requirement on the object (if **UNIQUE**, or group of objects if **NON_UNIQUE**) to have a specified property (\eg *pickupable*),
- **RELATION_UNIQUE** - requires objects connected by a relation to have a unique solution, \ie when one object is referred to in a vague way and in relation to second object, the combination has to provide a valid, unique solution
- **NOT_TP** - requires target object not to occupy given table part, both provided as target tokens, used to assure that when asking to move object to another table part, it is not already there
- **TP_SET** - requires the set of objects on the given table part to share given property, \eg all objects on the table part related to the given token have to be pickupable
- **IN** - requires given object target to be included in the given list of the second target, e.g. if one created a field in object structure with the list of objects that can be contained within

## Example instructions
We provide some example instructions corresponding to tasks we proposed in SHOP-VRB2:
- *Measure the weight of the OBJ1.* - Measure the weight of the yellow plastic object. 
- *What is the weight of all OBJ1s?* - What is the weight of all blue objects?. 
- *Pick up the WS1 of all OBJ1s.* - Pick up the heaviest of all bowls. 
- *Place the OBJ1 on the TP1 part of the table.* - Place the green mug on the left part of the table. 
- *Remove all OBJ1s from the TP1 part of the table.* - Remove all plastic objects from the right part of the table. 
- *Place the WS1 of all OBJ1s on the TP1 part of the table.* - Place the lightest of all prismatic objects on the left part of the table. 
- *Stack the OBJ1 on top of the OBJ2.* - Stack the cylindrical object on top of the box. 
- *Place the WS1 of all OBJ1s on top of the OBJ2.* - Place the heaviest of all hemispheres on top of the green object.
- *Stack the OBJ1 on top of the OBJ2 on top of the OBJ3.* - Stack the mug on top of the red bowl on top of the prismatic object. 
- *Stack all OBJ1s from heaviest to lightest.* - Stack all metal objects from heaviest to lightest. 
