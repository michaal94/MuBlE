'''
Based on:
clevr-dataset-gen
https://github.com/facebookresearch/clevr-dataset-gen
'''

from __future__ import print_function
import os
import sys
# As its called from blender we need to add the following
# to import from inside project
PROJECT_PATH = os.path.abspath('..')
sys.path.insert(0, PROJECT_PATH)
print("Python paths:")
print(sys.path)
import random
import argparse
import json
import tempfile
import pycocotools.mask as mask_utils
from collections import Counter
import cv2
import numpy as np
from mathutils.bvhtree import BVHTree

"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information, such as given properties and encoded segmentation mask.

This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""


INSIDE_BLENDER = True
try:
    import bpy
    from mathutils import Vector
except ImportError:
    INSIDE_BLENDER = False
if INSIDE_BLENDER:
    try:
        import scene_generation.utils as utils
        from scene_generation.config import Config
    except ImportError:
        print("\nERROR")
        print("Running render_images.py from Blender and cannot import utils.py.")
        print("You may need to add a .pth file to the site-packages of Blender's")
        print("bundled python with a command like this:\n")
        print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.7/site-packages/shop_vrb.pth")
        print("\nWhere $BLENDER is the directory where Blender is installed, and")
        print("$VERSION is your Blender version (such as 2.81).")
        sys.exit(1)

parser = argparse.ArgumentParser()

# Config file
parser.add_argument(
    "--config", "-c",
    default=os.path.join("configs", "default.json")
)

def main(args):
    unknown_args = args[1]
    args = args[0]
    config = Config()
    config.merge_from_json(args.config)
    config.merge_from_cmd(unknown_args)
    print(config)
    num_digits = 6
    prefix = '%s_%s_' % (config.filename_prefix, config.split)
    img_template = '%s%%0%dd.png' % (prefix, num_digits)
    scene_template = '%s%%0%dd.json' % (prefix, num_digits)
    blend_template = '%s%%0%dd.blend' % (prefix, num_digits)
    img_template = os.path.join(config.output_image_dir, img_template)
    scene_template = os.path.join(config.output_scene_dir, scene_template)
    blend_template = os.path.join(config.output_blend_dir, blend_template)

    if not os.path.isdir(config.output_image_dir):
        os.makedirs(config.output_image_dir)
    if not os.path.isdir(config.output_scene_dir):
        os.makedirs(config.output_scene_dir)
    if config.save_blendfiles and not os.path.isdir(config.output_blend_dir):
        os.makedirs(config.output_blend_dir)

    all_scene_paths = []
    for i in range(config.num_images):
        img_path = img_template % (i + config.start_idx)
        scene_path = scene_template % (i + config.start_idx)
        all_scene_paths.append(scene_path)
        blend_path = None
        if config.save_blendfiles:
            blend_path = blend_template % (i + config.start_idx)
        num_objects = random.randint(config.min_objects, config.max_objects)
        render_scene(
            config,
            num_objects=num_objects,
            output_index=(i + config.start_idx),
            output_split=config.split,
            output_image=img_path,
            output_scene=scene_path,
            output_blendfile=blend_path,
        )

    # After rendering all images, combine the JSON files for each scene into a
    # single JSON file.
    all_scenes = []
    for scene_path in all_scene_paths:
        with open(scene_path, 'r') as f:
            all_scenes.append(json.load(f))
    output = {
        'info': {
            'date': config.date,
            'version': config.version,
            'split': config.split,
            'license': config.license,
        },
        'scenes': all_scenes
    }
    with open(config.output_scene_file, 'w') as f:
        json.dump(output, f)


def render_scene(
        config,
        num_objects=5,
        output_index=0,
        output_split='none',
        output_image='render.png',
        output_scene='render_json',
        output_blendfile=None):

    # Load the main blendfile
    bpy.ops.wm.open_mainfile(filepath=config.base_scene_blendfile)

    # Load materials
    utils.load_materials(config.material_dir)

    # Set render arguments so we can get pixel coordinates later.
    # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
    # cannot be used.
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    render_args.filepath = output_image
    render_args.resolution_x = config.width
    render_args.resolution_y = config.height
    render_args.resolution_percentage = 100
    render_args.tile_x = config.render_tile_size
    render_args.tile_y = config.render_tile_size
    if config.use_gpu:
        # Blender changed the API for enabling CUDA at some point
        if bpy.app.version < (2, 78, 0):
            bpy.context.user_preferences.system.compute_device_type = 'CUDA'
            bpy.context.user_preferences.system.compute_device = 'CUDA_0'
        elif bpy.app.version < (2, 80, 0):
            cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
            cycles_prefs.compute_device_type = 'CUDA'
        else:
            cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
            if config.use_optix:
                cycles_prefs.compute_device_type = 'OPTIX'
            else:
                cycles_prefs.compute_device_type = 'CUDA'
            bpy.context.preferences.addons["cycles"].preferences.get_devices()

    if not (config.use_gpu and config.use_optix):
        bpy.data.worlds['World'].cycles.sample_as_light = True

    # Some CYCLES-specific stuff
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = config.render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = config.render_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = config.render_max_bounces
    if config.use_gpu:
        bpy.context.scene.cycles.device = 'GPU'

    if config.active_cam == 0:
        active_camera = 'cam0'
    elif config.active_cam == 1:
        active_camera = 'cam1'
    else:
        active_camera = random.choice(['cam0', 'cam1'])

    bpy.context.scene.camera = bpy.data.objects[active_camera]

    # This will give ground-truth information about the scene and its objects
    scene_struct = {
        'split': output_split,
        'image_index': output_index,
        'image_filename': os.path.basename(output_image),
        'objects': [],
        'directions': {},
    }

    # Put a plane on the ground so we can compute cardinal directions
    bpy.ops.mesh.primitive_plane_add(size=5)
    plane = bpy.context.object

    def rand(L):
        return 2.0 * L * (random.random() - 0.5)

    # Add random jitter to camera position
    # if args.camera_jitter > 0:
    #     for i in range(3):
    #         bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)

    # Figure out the left, up, and behind directions along the plane and record
    # them in the scene structure
    # camera = bpy.data.objects['Camera']
    camera = bpy.data.objects[active_camera]
    plane_normal = plane.data.vertices[0].normal
    cam_behind = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
    cam_left = camera.matrix_world.to_quaternion() @ Vector((-1, 0, 0))
    cam_up = camera.matrix_world.to_quaternion() @ Vector((0, 1, 0))
    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    plane_up = cam_up.project(plane_normal).normalized()

    # Delete the plane; we only used it for normals anyway. The base scene file
    # contains the actual ground plane.
    utils.delete_object(plane)

    scene_struct['camera_params'] = {}
    scene_struct['camera_params']['position'] = list(camera.location)
    cam_rot = [np.rad2deg(x) for x in list(camera.rotation_euler)]
    scene_struct['camera_params']['rotation'] = cam_rot
    scene_struct['camera_params']['rotation_mode'] = 'EulerXYZ'
    # print(scene_struct['camera_params'])

    # exit()

    # Save all six axis-aligned directions in the scene struct
    scene_struct['directions']['behind'] = tuple(plane_behind)
    scene_struct['directions']['front'] = tuple(-plane_behind)
    scene_struct['directions']['left'] = tuple(plane_left)
    scene_struct['directions']['right'] = tuple(-plane_left)
    scene_struct['directions']['above'] = tuple(plane_up)
    scene_struct['directions']['below'] = tuple(-plane_up)

    # Add random jitter to lamp positions
    if config.key_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Key'].location[i] += rand(config.key_light_jitter)
    if config.back_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Back'].location[i] += rand(config.back_light_jitter)
    if config.fill_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Fill'].location[i] += rand(config.fill_light_jitter)

    scene_struct['lamp_params'] = {
        'Lamp_Key': list(bpy.data.objects['Lamp_Key'].location),
        'Lamp_Back': list(bpy.data.objects['Lamp_Back'].location),
        'Lamp_Fill': list(bpy.data.objects['Lamp_Fill'].location)
    }

    if 'Robot' in bpy.data.collections:
        for obj in bpy.data.collections['Robot'].all_objects:
            if obj.parent is None:
                pos, quat = config.robot_init_config[obj.name]
                obj.location = pos
                obj.rotation_mode = 'QUATERNION'
                obj.rotation_quaternion = quat

    if 'Gripper' in bpy.data.collections:
        for obj in bpy.data.collections['Gripper'].all_objects:
            if obj.parent is None:
                pos, quat = config.gripper_init_config[obj.name]
                obj.location = pos
                obj.rotation_mode = 'QUATERNION'
                obj.rotation_quaternion = quat

    # Now make some random objects
    objects, blender_objects = None, None
    while objects is None:
        objects, blender_objects = add_random_objects(scene_struct, num_objects, config, camera)
        print("\n\nRelocating all objects\n\n")

    # bpy.context.scene.node_tree.nodes["Switch"].check = False
    # bpy.context.view_layer.cycles.use_denoising = False

    # Render the scene and dump the scene data structure
    scene_struct['objects'] = objects
    scene_struct['relationships'] = compute_all_relationships(scene_struct)
    while True:
        try:
            bpy.ops.render.render(write_still=True)
            break
        except Exception as e:
            print(e)

    with open(output_scene, 'w') as f:
        json.dump(scene_struct, f, indent=2)

    if output_blendfile is not None:
        bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)


def add_random_objects(scene_struct, num_objects, config, camera):
    """
    Add random objects to the current blender scene
    """

    # Load the property file
    with open(config.properties_json, 'r') as f:
        properties = json.load(f)
        color_name_to_rgba = {}
        for name, rgb in properties['colors'].items():
            rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            color_name_to_rgba[name] = rgba
        object_mapping = [(v, k) for k, v in properties['shapes'].items()]
        size_mapping = properties['sizes']

    white = [0.88, 0.88, 0.88, 1.0]

    with open(config.object_props, 'r') as f:
        object_properties = json.load(f)

    positions = []
    weights = []
    objects = []
    blender_objects = []
    bvh_list = []

    if 'Robot' in bpy.data.collections:
        correct = True
        if 'link0' in bpy.data.objects:
            obj = bpy.data.objects['link0']
        elif 'base_link' in bpy.data.objects:
            obj = bpy.data.objects['base_link']
        else:
            correct = False
        if correct:
            target_mat = obj.matrix_world
            target_vert = [target_mat @ v.co for v in obj.data.vertices]
            target_poly = [p.vertices for p in obj.data.polygons]
            target_bvh = BVHTree.FromPolygons(target_vert, target_poly)
            bvh_list.append(target_bvh)

    for i in range(num_objects):
        # Choose random color and shape
        obj_name, obj_name_out = random.choice(object_mapping)
        if 'soda_can' in obj_name_out:
            obj_name_out = 'soda_can'
        color_name, rgba = random.choice(list(color_name_to_rgba.items()))

        # Choose a random size
        if object_properties[obj_name]['change_size']:
            size_name = random.choice(list(size_mapping.keys()))
            r = size_mapping[size_name]
            if size_name == "bigger":
                size_name = object_properties[obj_name]['size1']
            else:
                size_name = object_properties[obj_name]['size2']
        else:
            size_name = object_properties[obj_name]['size1']
            r = size_mapping["bigger"]

        num_tries = 0
        while True:
            # If we try and fail to place an object too many times, then delete all
            # the objects in the scene and start over.
            num_tries += 1
            # print(num_tries)
            if num_tries > config.max_retries:
                for obj in blender_objects:
                    utils.delete_object(obj)
                del blender_objects
                del objects
                return None, None
            x = random.uniform(-0.25, 0.25)
            x = random.uniform(0.3, 0.7)
            if config.active_cam == 0:
                y = random.uniform(-0.25, 0.25)
            elif config.active_cam == 1:
                y = random.uniform(-0.33, 0.11)
            else:
                y = random.uniform(-0.11, 0.33)


            # x = x + 0.4
            # y = y - 0.1

            # Choose random orientation for the object.
            theta = 180.0 * random.random()

            # Check to make sure the new object is further than min_dist from all
            # other objects, and further than margin along the four cardinal directions
            margins_good = True

            pos_temp = positions.copy()
            pos_temp.append((x, y, r, theta))

            for (xx, yy, rr, th) in positions:
                dx, dy = x - xx, y - yy
                for direction_name in ['left', 'right', 'front', 'behind']:
                    direction_vec = scene_struct['directions'][direction_name]
                    assert direction_vec[2] == 0
                    margin = dx * direction_vec[0] + dy * direction_vec[1]
                    if 0 < margin < config.margin:
                        print(margin, config.margin, direction_name)
                        print('Broken margin!')
                        margins_good = False
                        break
                if not margins_good:
                    break

            # Actually add the object to the scene
            utils.add_object(config.shape_dir, obj_name, r, (x, y), theta=theta)
            obj = bpy.context.object

            dists_good, new_bvh = utils.check_intersections_bvh(bvh_list, obj, config.intersection_eps)
            dists_good = not dists_good

            if dists_good and margins_good:
                bvh_list.append(new_bvh)
                break
            else:
                utils.delete_object(obj)

        blender_objects.append(obj)
        positions.append((x, y, r, theta))

        # Attach a random material
        if object_properties[obj_name]['change_material']:
            change = random.choice([True, False])
            if change:
                mat_name_out = object_properties[obj_name]['material2']
                mat_name = mat_name_out.capitalize()
                if object_properties[obj_name]['change_color2']:
                    utils.add_material(mat_name, Color=rgba)
                else:
                    color_name = object_properties[obj_name]['color2']
                    utils.add_material(mat_name, Color=white)
            else:
                mat_name_out = object_properties[obj_name]['material1']
                if object_properties[obj_name]['change_color1']:
                    utils.add_color(rgba)
                else:
                    color_name = object_properties[obj_name]['color1']
        else:
            mat_name_out = object_properties[obj_name]['material1']
            if object_properties[obj_name]['change_color1']:
                print(obj_name)
                utils.add_color(rgba)
            else:
                color_name = object_properties[obj_name]['color1']

        # Record data about the object in the scene data structure
        pixel_coords = utils.get_camera_coords(camera, obj.location)

        obj_name_out = object_properties[obj_name]['name']

        weight = object_properties[obj_name]['weight_gt_mean'] * r
        for _ in range(100):
            rand_percentage = np.random.uniform(-config.weight_range, config.weight_range)
            weight_out = (1 + rand_percentage) * weight
            weight_out = np.around(weight_out, decimals=1)
            if weight_out not in weights:
                weights.append(weight_out)
                break
        stackable = object_properties[obj_name]['stackable']
        stack_base = object_properties[obj_name]['stack_base']
        pickupable = object_properties[obj_name]['pickupable']

        movability = object_properties[obj_name]['movability']
        shape = object_properties[obj_name]['shape']

        x, y, z = obj.dimensions
        bbox = {
            'x': x,
            'y': y,
            'z': z
        }
        obj.rotation_mode = 'QUATERNION'

        objects.append({
            'file': obj_name,
            'name': obj_name_out,
            'shape': shape,
            'size': size_name,
            'material': mat_name_out,
            '3d_coords': tuple(obj.location),
            'orientation': tuple(obj.rotation_quaternion),
            'rotation': theta,
            'pixel_coords': pixel_coords,
            'colour': color_name,
            'weight_gt': weight_out,
            'weight': None,
            'movability': movability,
            'bbox': bbox,
            'scale_factor': r,
            'stackable': stackable,
            'stack_base': stack_base,
            'pickupable': pickupable
        })

    # Check that all objects are at least partially visible
    # in the rendered image
    all_visible, masks, extra_masks = check_visibility(blender_objects, config.min_pixels_per_object)
    if not all_visible:
        # If any of the objects are fully occluded then start over; delete all
        # objects from the scene and place them all again.
        print('Some objects are occluded; replacing objects')
        for obj in blender_objects:
            utils.delete_object(obj)
        return add_random_objects(scene_struct, num_objects, config, camera)

    for obj, mask in zip(objects, masks):
        obj['mask'] = mask

    if 'table' in extra_masks:
        scene_struct['table'] = {}
        scene_struct['table']['mask'] = extra_masks['table']

    if 'robot' in extra_masks:
        scene_struct['robot'] = {}
        scene_struct['robot']['mask'] = extra_masks['robot']

    return objects, blender_objects


def compute_all_relationships(scene_struct, eps=0.0):
    """
    Computes relationships between all pairs of objects in the scene.

    Returns a dictionary mapping string relationship names to lists of lists of
    integers, where output[rel][i] gives a list of object indices that have the
    relationship rel with object i. For example if j is in output['left'][i] then
    object j is left of object i.
    """
    all_relationships = {}
    for name, direction_vec in scene_struct['directions'].items():
        if name == 'above' or name == 'below':
            continue
        all_relationships[name] = []
        for i, obj1 in enumerate(scene_struct['objects']):
            coords1 = obj1['3d_coords']
            related = set()
            for j, obj2 in enumerate(scene_struct['objects']):
                if obj1 == obj2:
                    continue
                coords2 = obj2['3d_coords']
                diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
                dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
                if dot > eps:
                    related.add(j)
            all_relationships[name].append(sorted(list(related)))
    return all_relationships


def check_visibility(blender_objects, min_pixels_per_object):
    """
    Check whether all objects in the scene have some minimum number of visible
    pixels; to accomplish this we assign random (but distinct) colors to all
    objects, and render using no lighting or shading or antialiasing; this
    ensures that each object is just a solid uniform color. We can then count
    the number of pixels of each color in the output image to check the visibility
    of each object.

    Returns True if all objects are visible and False otherwise.
    If True returns segmentation mask as well
    """
    f, path = tempfile.mkstemp(suffix='.png')
    # path = '../output/test_mask.png'
    # Render shadeless and return list of colours
    object_colors = render_shadeless(blender_objects, path)
    img = bpy.data.images.load(path)
    p = list(img.pixels)
    # Count whether the number of colours is correct - full occlusion
    color_count = Counter((p[i], p[i + 1], p[i + 2], p[i + 3]) for i in range(0, len(p), 4))
    additional_objects = 1
    # print(color_count)
    # exit()
    if 'Desk' in bpy.data.objects:
        additional_objects += 1
    if 'Robot' in bpy.data.collections:
        additional_objects += 1
    if len(color_count) != len(blender_objects) + additional_objects:
        print("Full occlusion detected")
        return False, None, None
    # Check partial occlusion
    for col, count in color_count.most_common():
        # print(col, count)
        if count < min_pixels_per_object:
            print("Partial occlusion detected")
            return False, None, None

    # Assign masks
    masks = assign_masks(object_colors, path)
    extra_masks = dict()
    if 'Desk' in bpy.data.objects:
        desk_mask = assign_masks([(80, 80, 80)], path)
        if desk_mask is not None:
            extra_masks['table'] = desk_mask[0]
    if 'Robot' in bpy.data.collections:
        robot_mask = assign_masks([(127, 127, 127)], path)
        if robot_mask is not None:
            extra_masks['robot'] = robot_mask[0]
    return True, masks, extra_masks


def render_shadeless(blender_objects, path='flat.png'):
    """
    Render a version of the scene with shading disabled and unique materials
    assigned to all objects, and return a set of all colors that should be in the
    rendered image. The image itself is written to path. This is used to ensure
    that all objects will be visible in the final rendered scene.
    """
    render_args = bpy.context.scene.render

    # Cache the render args we are about to clobber
    old_filepath = render_args.filepath
    old_filter_size = render_args.filter_size

    # Override some render settings to have flat shading
    render_args.filepath = path

    # Switch denoising state
    old_denoising_state = bpy.context.scene.node_tree.nodes["Switch"].check
    bpy.context.scene.node_tree.nodes["Switch"].check = False
    old_cycles_denoising = bpy.context.view_layer.cycles.use_denoising
    bpy.context.view_layer.cycles.use_denoising = False

    # Don't render lights
    utils.set_render(bpy.data.objects['Lamp_Key'], False)
    utils.set_render(bpy.data.objects['Lamp_Fill'], False)
    utils.set_render(bpy.data.objects['Lamp_Back'], False)
    utils.set_render(bpy.data.objects['Ground'], False)

    # Change shading and AA
    old_shading = bpy.context.scene.display.shading.light
    bpy.context.scene.display.shading.light = 'FLAT'
    old_aa = bpy.context.scene.display.render_aa
    bpy.context.scene.display.render_aa = 'OFF'

    # Cycles settings
    old_blur = bpy.context.scene.cycles.blur_glossy
    bpy.context.scene.cycles.blur_glossy = 0.0
    old_samples = bpy.context.scene.cycles.samples
    bpy.context.scene.cycles.samples = 1
    old_light_bounces = bpy.context.scene.cycles.max_bounces
    bpy.context.scene.cycles.max_bounces = 0

    # Add random shadeless materials to all objects
    object_colors = []
    new_obj = []

    def create_shadeless_copy(obj, num, colour):
        obj.select_set(state=True)
        bpy.ops.object.duplicate(linked=False, mode='INIT')
        utils.set_render(obj, False)
        mat = bpy.data.materials['Shadeless'].copy()
        mat.name = 'Shadeless_temp_%d' % num
        group_node = mat.node_tree.nodes['Group']
        r, g, b = colour
        for inp in group_node.inputs:
            if inp.name == 'Color':
                inp.default_value = (float(r) / 255, float(g) / 255, float(b) / 255, 1.0)
        for i in range(len(bpy.context.selected_objects[0].data.materials)):
            bpy.context.selected_objects[0].data.materials[i] = mat

    for obj in bpy.data.objects:
        obj.select_set(state=False)
    for i, obj in enumerate(blender_objects):
        while True:
            r, g, b = [random.randint(0, 255) for _ in range(3)]
            colour_correct = True
            colour_correct = colour_correct and ((r, g, b) not in object_colors)
            colour_correct = colour_correct and ((r, g, b) != (13, 13, 13))
            colour_correct = colour_correct and ((r, g, b) != (80, 80, 80))
            colour_correct = colour_correct and ((r, g, b) != (127, 127, 127))
            if colour_correct:
                break
        object_colors.append((r, g, b))
        create_shadeless_copy(obj, i, (r, g, b))
        new_obj.append(bpy.context.selected_objects[0])
        for o in bpy.data.objects:
            o.select_set(state=False)

    if 'Desk' in bpy.data.objects:
        i = i + 1
        create_shadeless_copy(bpy.data.objects['Desk'], i, (80, 80, 80))
        new_obj.append(bpy.context.selected_objects[0])
        for o in bpy.data.objects:
            o.select_set(state=False)

    if 'Robot' in bpy.data.collections:
        robot_parts = []
        for obj in bpy.data.collections['Robot'].all_objects:
            robot_parts.append(obj)

        for r_part in robot_parts:
            i = i + 1
            create_shadeless_copy(r_part, i, (127, 127, 127))
            new_obj.append(bpy.context.selected_objects[0])
            for o in bpy.data.objects:
                o.select_set(state=False)

    if 'Gripper' in bpy.data.collections:
        gripper_parts = []
        for obj in bpy.data.collections['Gripper'].all_objects:
            gripper_parts.append(obj)

        for g_part in gripper_parts:
            i = i + 1
            create_shadeless_copy(g_part, i, (127, 127, 127))
            new_obj.append(bpy.context.selected_objects[0])
            for o in bpy.data.objects:
                o.select_set(state=False)

    # Render the scene
    # Save gamma
    gamma = bpy.context.scene.view_settings.view_transform
    bpy.context.scene.view_settings.view_transform = 'Raw'
    bpy.ops.render.render(write_still=True)
    bpy.context.scene.view_settings.view_transform = gamma

    # Undo the above; first restore the materials to objects
    for obj in new_obj:
        obj.select_set(state=True)
        bpy.ops.object.delete()

    for obj in blender_objects:
        utils.set_render(obj, True)

    if 'Desk' in bpy.data.objects:
        utils.set_render(bpy.data.objects['Desk'], True)

    if 'Robot' in bpy.data.collections:
        for r_part in robot_parts:
            utils.set_render(r_part, True)

    if 'Gripper' in bpy.data.collections:
        for g_part in gripper_parts:
            utils.set_render(g_part, True)

    # Render lights again
    utils.set_render(bpy.data.objects['Lamp_Key'], True)
    utils.set_render(bpy.data.objects['Lamp_Fill'], True)
    utils.set_render(bpy.data.objects['Lamp_Back'], True)
    utils.set_render(bpy.data.objects['Ground'], True)

    # Set the render settings back to what they were
    render_args.filepath = old_filepath
    render_args.filter_size = old_filter_size
    bpy.context.scene.display.shading.light = old_shading
    bpy.context.scene.display.render_aa = old_aa
    bpy.context.scene.node_tree.nodes["Switch"].check = old_denoising_state
    bpy.context.view_layer.cycles.use_denoising = old_cycles_denoising
    bpy.context.scene.cycles.blur_glossy = old_blur
    bpy.context.scene.cycles.samples = old_samples
    bpy.context.scene.cycles.max_bounces = old_light_bounces

    return object_colors


def assign_masks(colors, image_path):
    masks = []
    img = cv2.imread(image_path)
    img_rgb = img[:, :, [2, 1, 0]]
    # Read shadeless render and encode masks in COCO format
    for color in colors:
        mask = np.all(img_rgb == color, axis=-1)
        mask = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        mask['counts'] = str(mask['counts'], "utf-8")
        masks.append(mask)
    return masks


if __name__ == '__main__':
    if INSIDE_BLENDER:
        # Run normally
        argv = utils.extract_args()
        known_args, unknown_args = parser.parse_known_args(argv)
        main((known_args, unknown_args))
    elif '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
    else:
        print('This script is intended to be called from blender like this:')
        print()
        print('blender --background --python render_images.py -- [args]')
        print()
        print('You can also run as a standalone python script to view all')
        print('arguments like this:')
        print()
        print('python render_images.py --help')
