import os
import json
from datetime import datetime as dt


class Config:
    def __init__(self) -> None:
        # DATA SOURCE PARAMETERS

        # Base blender file on which all scenes are based; 
        # includes ground plane, lights, and camera.
        self.base_scene_blendfile = 'data/base_scene.blend'
        # JSON file defining objects, materials, sizes, and colors. 
        # The \"colors\" field maps from color names to RGB values;
        # The \"sizes\" field maps from size names to scalars used to
        # rescale object models; the \"materials\" and \"shapes\" fields map
        # from material and shape names to .blend files in the
        # --object_material_dir and --shape_dir directories respectively.
        self.properties_json = 'data/properties.json'
        # JSON file defining properties of objects.
        # Properties are ground truth information about the objects 
        # along with indication whether material, colour or size 
        # should be changed.
        self.object_props = 'data/object_properties.json'
        # Directory where .blend files for object models are stored
        self.shape_dir = 'data/shapes'
        # Directory where .blend files for materials are stored
        self.material_dir = 'data/materials'

        # OBJECT LOCATION PARAMETERS
        # The minimum number of objects to place in each scene
        self.min_objects = 3
        # The maximum number of objects to place in each scene 
        self.max_objects = 5
        # The minimum allowed distance between object centers
        self.min_dist = 0.05
        # Fraction of weight to be randomised
        self.weight_range = 0.05
        # All objects will have at least this many visible pixels in the
        # final rendered images; this ensures that no objects are fully
        # occluded by other objects.
        self.min_pixels_per_object = 200
        # The number of times to try placing an object before giving up and
        # re-placing all objects in the scene.
        self.max_retries = 50
        # Minimum distance between the closest points between two meshes (objects).
        self.intersection_eps = 0.05
        # Along all cardinal directions (left, right, front, back), all 
        # objects will be at least this distance apart. This makes resolving 
        # spatial relationships slightly less ambiguous.
        self.margin = 0.04

        # OUTPUT SETTINGS
        # The index at which to start for numbering rendered images. 
        # Setting this to non-zero values allows you to distribute rendering 
        # across multiple machines and recombine the results later.
        self.start_idx = 0
        # The number of images to render
        self.num_images = 5
        # This prefix will be prepended to the rendered images and JSON scenes
        self.filename_prefix = 'NS_AP'
        # Name of the split for which we are rendering. 
        # This will be added to the names of rendered images,
        # and will also be stored in the JSON scene structure for each image.
        self.split = 'train'
        # The directory where output images will be stored. 
        # It will be created if it does not exist.
        self.output_image_dir = '../output/images/'
        # The directory where output JSON scene structures will be stored. 
        # It will be created if it does not exist.
        self.output_scene_dir = '../output/scenes/'
        # Path to write a single JSON file containing all scene information.
        self.output_scene_file = '../output/NS_AP_scenes.json'
        # The directory where blender scene files will be stored, 
        # if the user requested that these files be saved using the
        # --save_blendfiles flag; in this case it will be created 
        # if it does not already exist.
        self.output_blend_dir = '../output/blendfiles'
        # Setting --save_blendfiles will cause the blender scene file for
        # each generated image to be stored in the directory specified by
        # the --output_blend_dir flag. These files are not saved by default
        # because they take up a lot of space.
        self.save_blendfiles = False
        # String to store in the \"version\" field of the generated JSON file
        self.version = '1.0'
        # String to store in the \"license\" field of the generated JSON file"
        self.license = "Creative Commons Attribution (CC-BY 4.0)"
        # String to store in the \"date\" field of the generated JSON file;
        # defaults to today's date
        self.date = dt.today().strftime("%d/%m/%Y")

        # RENDERING OPTIONS
        # Setting --use_gpu enables GPU-accelerated rendering using CUDA.
        # You must have an NVIDIA GPU with the CUDA toolkit installed for it to work.
        self.use_gpu = False
        # Setting --use_optix enables GPU-accelerated rendering using OptiX.
        # You must have an NVIDIA RTX with Nvidia drivers >=435
        # to work. Faster than CUDA (slightly). Must set use_gpu before.
        self.use_optix = False
        # The width (in pixels) for the rendered images
        self.width = 848
        # The height (in pixels) for the rendered images
        self.height = 480
        # The magnitude of random jitter to add to the key light position.
        self.key_light_jitter = 0.05
        # The magnitude of random jitter to add to the fill light position.
        self.fill_light_jitter = 0.05
        # The magnitude of random jitter to add to the back light position.
        self.back_light_jitter = 0.05
        # The magnitude of random jitter to add to the camera position.
        # Zero default cause we are replicating real setup
        self.camera_jitter = 0.0
        # The number of samples to use when rendering. Larger values will
        # result in nicer images but will cause rendering to take longer.
        # No need to be that high since Blender 2.8 which has built-in denoiser.
        self.render_num_samples = 64
        # The minimum number of bounces to use for rendering.
        self.render_min_bounces = 8
        # The maximum number of bounces to use for rendering.
        self.render_max_bounces = 8
        # The tile size to use for rendering. This should not affect the
        # quality of the rendered image but may affect the speed; CPU-based
        # rendering may achieve better performance using smaller tile sizes
        # while larger tile sizes may be optimal for GPU-based rendering. 
        # If it works with your GPU you can set it even to size covering the whole image.
        self.render_tile_size = 2048
        # We replicate setup with 2 cameras, the following parameter choses one
        self.active_cam =-1

        ## ROBOT PARAMETERS
        self.robot_init_config = {
            'link0': ([0., 0., 0.], [1., 0., 0., 0.]), 
            'link1': ([0.   , 0.   , 0.333], [ 0.99996686,  0.        ,  0.        , -0.00814159]), 
            'link2': ([0.   , 0.   , 0.333], [ 0.70709987, -0.70705704,  0.00838687, -0.003127  ]),
            'link3': ([ 2.35034211e-03, -3.82748638e-05,  6.48991257e-01], [ 9.99982226e-01,  4.32315698e-05,  3.71916331e-03, -4.65983623e-03]),
            'link4': ([ 0.08484448, -0.00080711,  0.64837757], [ 0.27288862,  0.27897062,  0.64977019, -0.65230813]),
            'link5': ([ 0.41813879, -0.00390273,  0.44060951], [3.90227048e-01, 9.98689328e-03, 9.20664290e-01, 6.15478348e-04]),
            'link6': ([ 0.41813879, -0.00390273,  0.44060951], [ 0.70392049,  0.71001676, -0.01843918,  0.00566933]),
            'link7': ([ 0.50607329, -0.00550457,  0.4436024 ], [-0.01058263,  0.91810927, -0.39593709,  0.01404281])
        }
        self.gripper_init_config = {
            'robotiq_85_adapter_link': ([ 0.50971195, -0.00461935,  0.33716826], [-0.00439893,  0.99974446, -0.01420672,  0.01702474]),
            'left_inner_finger': ([0.51498114, 0.0655301 , 0.24102349], [-0.01898179, -0.01146143, -0.98752135, -0.15591636]),
            'left_inner_knuckle': ([0.51216927, 0.00858555, 0.27590409], [-0.01836344, -0.01242807, -0.99426502, -0.10462027]),
            'left_outer_knuckle': ([0.51245246, 0.02642457, 0.28258217], [-0.01721945, -0.01397008, -0.99958819, -0.018215  ]),
            'right_inner_finger': ([ 0.51106056, -0.07315872,  0.23973589], [ 0.14718429,  0.98886051, -0.01662304,  0.01467472]),
            'right_inner_knuckle': ([ 0.51145155, -0.01680322,  0.27566839], [ 0.09586795,  0.99514706, -0.0158422 ,  0.01551443]),
            'right_outer_knuckle': ([ 0.51072311, -0.03475056,  0.28201425], [ 0.00941682,  0.99970978, -0.01444062,  0.0168268 ])
        }


    def merge_from_json(self, json_path):
        with open(json_path, 'r') as f:
            json_config = json.load(f)
        self.merge_from_dict(json_config)

    def merge_from_cmd(self, args_list):
        self_args = vars(self)
        args_dict = dict()
        for i in range(len(args_list)):
            arg = args_list[i]
            if "--" in arg:
                key = arg.strip("-")
                if (i + 1) < len(args_list) and not ("--" in args_list[i + 1]):
                    value = args_list[i + 1]
                else:
                    value = True
                if isinstance(self_args[key], int):
                    value = int(value)
                if isinstance(self_args[key], float):
                    value = float(value)
                args_dict[key] = value
        self.merge_from_dict(args_dict)

    def merge_from_dict(self, args_dict):
        if "base_scene_blendfile" in args_dict:
            self.base_scene_blendfile = args_dict["base_scene_blendfile"]
        if "properties_json" in args_dict:
            self.properties_json = args_dict["properties_json"]
        if "object_props" in args_dict:
            self.object_props = args_dict["object_props"]
        if "shape_dir" in args_dict:
            self.shape_dir = args_dict["shape_dir"]
        if "material_dir" in args_dict:
            self.material_dir = args_dict["material_dir"]
        if "min_objects" in args_dict:
            self.min_objects = args_dict["min_objects"]
        if "max_objects" in args_dict:
            self.max_objects = args_dict["max_objects"]
        if "min_dist" in args_dict:
            self.min_dist = args_dict["min_dist"]
        if "min_pixels_per_object" in args_dict:
            self.min_pixels_per_object = args_dict["min_pixels_per_object"]
        if "max_retries" in args_dict:
            self.max_retries = args_dict["max_retries"]
        if "intersection_eps" in args_dict:
            self.intersection_eps = args_dict["intersection_eps"]
        if "margin" in args_dict:
            self.margin = args_dict["margin"]
        if "start_idx" in args_dict:
            self.start_idx = args_dict["start_idx"]
        if "num_images" in args_dict:
            self.num_images = args_dict["num_images"]
        if "filename_prefix" in args_dict:
            self.filename_prefix = args_dict["filename_prefix"]
        if "split" in args_dict:
            self.split = args_dict["split"]
        if "output_image_dir" in args_dict:
            self.output_image_dir = args_dict["output_image_dir"]
        if "output_scene_dir" in args_dict:
            self.output_scene_dir = args_dict["output_scene_dir"]
        if "output_scene_file" in args_dict:
            self.output_scene_file = args_dict["output_scene_file"]
        if "output_blend_dir" in args_dict:
            self.output_blend_dir = args_dict["output_blend_dir"]
        if "save_blendfiles" in args_dict:
            self.save_blendfiles = args_dict["save_blendfiles"]
        if "version" in args_dict:
            self.version = args_dict["version"]
        if "license" in args_dict:
            self.license = args_dict["license"]
        if "date" in args_dict:
            self.date = args_dict["date"]
        if "use_gpu" in args_dict:
            self.use_gpu = args_dict["use_gpu"]
        if "use_optix" in args_dict:
            self.use_optix = args_dict["use_optix"]
        if "width" in args_dict:
            self.width = args_dict["width"]
        if "height" in args_dict:
            self.height = args_dict["height"]
        if "key_light_jitter" in args_dict:
            self.key_light_jitter = args_dict["key_light_jitter"]
        if "fill_light_jitter" in args_dict:
            self.fill_light_jitter = args_dict["fill_light_jitter"]
        if "back_light_jitter" in args_dict:
            self.back_light_jitter = args_dict["back_light_jitter"]
        if "camera_jitter" in args_dict:
            self.camera_jitter = args_dict["camera_jitter"]
        if "render_num_samples" in args_dict:
            self.render_num_samples = args_dict["render_num_samples"]
        if "render_min_bounces" in args_dict:
            self.render_min_bounces = args_dict["render_min_bounces"]
        if "render_max_bounces" in args_dict:
            self.render_max_bounces = args_dict["render_max_bounces"]
        if "render_tile_size" in args_dict:
            self.render_tile_size = args_dict["render_tile_size"]
        if "active_cam" in args_dict:
            self.active_cam = args_dict["active_cam"]

    def __str__(self):
        msg = ""
        for k, v in vars(self).items():
            msg += "{0:30}\t{1}\n".format(k, v)
        return msg