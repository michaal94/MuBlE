---
layout: default
title: Sequences
parent: Dataset
nav_order: 5
---

# Sequence structure

Here we present the content of `sequence.json` associated with the ground truth sequence:
```json
{
    "info": {
        "image_filename": string,
        "instruction": string,
        "task": {
            "task": string,
            "target": float / list(idx)
        }
    },
    "observations_gt": [
        {
            "robot": {
                "pos": vector3,
                "ori": quaternion,
                "gripper_action": float,
                "gripper_closed": bool,
                "weight_measurement": float,
                "grasped_obj_idx": null / int,
                "robot_body": {
                    "link0": [vector3, quaternion],
                    ...
                },
                "gripper_body": {
                    "adapter_link": [vector3, quaternion],
                    ...
                },
                "robot_mask": "mask in COCO format",
                "table_mask": "mask in COCO format"
            },
            "objects": [
                {
                    ...
                    Same as in scene structure
                    ...
                    "bbox": [8 * vector3],
                    "pos": vector3,
                    "ori": quaternion,
                    "in_hand": bool,
                    "raised": bool,
                    "approached": bool,
                    "gripper_over": bool
                },
                ...
            ]
        },
        ...
    ],
    "image_paths": list(path),
    "result": string
}
```