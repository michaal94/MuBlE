---
layout: default
title: Scenes
parent: Dataset
nav_order: 3
---

# Scene structure

Here we present a structure of the scene `.json` files.

```json
{
  "split": string,
  "image_index": int,
  "image_filename": string,
  "objects": [
    {
      "file": string,
      "name": string,
      "shape": string,
      "colour": string,
      "material": string,
      "3d_coords": vector3,
      "orientation": quaternion,
      "weight_gt": float,
      "weight": null,
      "bbox": {
        "x": float,
        "y": float,
        "z": float
      },
      "scale_factor": float,
      "stackable": bool,
      "stack_base": bool,
      "pickupable": bool,
      "mask": "mask in COCO format"
    },
    ...
  ],
  "camera_params": {
    "position": vector3,
    "rotation": vector3 (XYZ angles),
    "rotation_mode": "EulerXYZ"
  },
  "lamp_params": {
    "Lamp_Key": vector3,
    "Lamp_Back": vector3,
    "Lamp_Fill": vector3
  },
  "table": {
    "mask": "mask in COCO format"
  },
  "robot": {
    "mask": "mask in COCO format"
  }
}
```