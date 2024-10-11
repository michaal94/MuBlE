---
layout: default
title: Scene generation
parent: MuBlE
nav_order: 1
---

# Scene generation

Scene generation is performed by going to `scene_generation` directory and calling:
```bash
cd scene_generation
blender -b -noaudio --python render_imnages.py -- -c config.json
```
where `config.json` is a path to configuration file.

We present a default/sample configuration file below:
```json
{
    "base_scene_blendfile": "data/base_scene.blend",
    "properties_json": "data/properties.json",
    "object_props": "data/object_properties.json",
    "shape_dir": "data/shapes",
    "material_dir": "data/materials",
    "min_objects": 4,
    "max_objects": 5,
    "min_dist": 0.05,
    "min_pixels_per_object": 200,
    "max_retries": 50,
    "intersection_eps": 0.05,
    "margin": 0.04,
    "start_idx": 0,
    "num_images": 5,
    "filename_prefix": "NS_AP",
    "split": "train",
    "output_image_dir": "../output/images/",
    "output_scene_dir": "../output/scenes/",
    "output_scene_file": "../output/NS_AP_scenes.json",
    "output_blend_dir": "../output/blendfiles",
    "save_blendfiles": false,
    "version": "1.0",
    "license": "Creative Commons Attribution (CC-BY 4.0)",
    "use_gpu": true,
    "use_optix": false,
    "width": 848,
    "height": 480,
    "key_light_jitter": 0.05,
    "fill_light_jitter": 0.05,
    "back_light_jitter": 0.05,
    "camera_jitter": 0.0,
    "render_num_samples": 64,
    "render_min_bounces": 8,
    "render_max_bounces": 8,
    "render_tile_size": 2048,
    "active_cam": -1
}
```

## Chnaging the scene / Adding new objects
In order to change the scene, use a new robot for blender rendering you need to make new `.blend` file based on `scene_generation/data/base_scene.blend`. In the future releases, we plan to expand default robot base. 

To add new objects, you have to prepare new `.blend` files containing the objects. Note that if you want the colour of the object to be able to be changes, you need to expose the material colour input, and name the desired material of the object *Changeable*, for an example open `scene_generation/data/shapes/plate2.blend`. Furhter, prepare `object_properties.json` containing the properties of all added objects (like `scene_generation/data/object_properties_ns_ap.json`). Finally, prepare the file containing properties of used objects, available colours and sizes (like `scene_generation/data/properties_ns_ap.json`).

## Additional tools
Make note of `collect_scenes.py`. It is a tool that collects all the scenes from the given directory into one `.json` file. It is required by the instruction generation script.


