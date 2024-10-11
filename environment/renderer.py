import bpy
import os
import cv2
import json
import random
import tempfile
import numpy as np
from scene_generation import utils
import pycocotools.mask as mask_utils


class BlenderRenderer:
    def __init__(self, config) -> None:
        self.objects = None
        self.robot_objects = None
        self.gripper_objects = None
        bpy.ops.wm.open_mainfile(filepath=config['base_scene_file'])
        utils.load_materials(config['materials_dir'])
        render_args = bpy.context.scene.render
        render_args.engine = "CYCLES"
        render_args.resolution_x = config['width']
        render_args.resolution_y = config['height']
        render_args.resolution_percentage = 100
        render_args.tile_x = config['render_tile_size']
        render_args.tile_y = config['render_tile_size']
        if config['use_gpu']:
            cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
            if config['use_optix']:
                cycles_prefs.compute_device_type = 'OPTIX'
            else:
                cycles_prefs.compute_device_type = 'CUDA'
            bpy.context.preferences.addons["cycles"].preferences.get_devices()

        if not (config['use_gpu'] and config['use_optix']):
            bpy.data.worlds['World'].cycles.sample_as_light = True

        # Some CYCLES-specific stuff
        bpy.context.scene.cycles.blur_glossy = 2.0
        bpy.context.scene.cycles.samples = config['render_num_samples']
        bpy.context.scene.cycles.transparent_min_bounces = config['render_min_bounces']
        bpy.context.scene.cycles.transparent_max_bounces = config['render_max_bounces']
        if config['use_gpu']:
            bpy.context.scene.cycles.device = 'GPU'

        with open(config['properties_json'], 'r') as f:
            properties = json.load(f)
            color_name_to_rgba = {}
            for name, rgb in properties['colors'].items():
                rgba = [float(c) / 255.0 for c in rgb] + [1.0]
                color_name_to_rgba[name] = rgba

        with open(config['object_properties_json'], 'r') as f:
            self.object_properties = json.load(f)

        color_name_to_rgba['white'] = [0.88, 0.88, 0.88, 1.0]
        self.color_name_to_rgba = color_name_to_rgba
        self.shape_dir = config['shape_dir']

    def init_scene(self, scene):
        # Match camera to scene data
        bpy.context.scene.camera = bpy.data.objects['cam0']
        bpy.data.objects['cam0'].location = scene['camera_params']['position']
        if 'Euler' in scene['camera_params']['rotation_mode']:
            # print('asdf00')
            # exit()
            rotation_mode = scene['camera_params']['rotation_mode'][5:]
            bpy.data.objects['cam0'].rotation_mode = rotation_mode
            bpy.data.objects['cam0'].rotation_euler = [
                np.deg2rad(x) for x in scene['camera_params']['rotation']
            ]
        else:
            bpy.data.objects['cam0'].rotation_mode = 'Quaternion'
            bpy.data.objects['cam0'].rotation_euler = scene['camera_params']['rotation']

        if 'lamp_params' in scene:
            for k, v in scene['lamp_params'].items():
                bpy.data.objects[k].location = v
        
        self.objects = []
        self._add_objects(scene['objects'])

    def _add_objects(self, objects):
        for obj in objects:
            obj_fname = obj['file']
            self._append_obj_to_scene(
                obj_fname,
                obj['scale_factor'],
                obj['3d_coords'],
                obj['orientation']
            )
            blender_obj = bpy.context.object
            self.objects.append(blender_obj)
            if obj['material'] != self.object_properties[obj_fname]['material1']:
                mat_name = obj['material'].capitalize()
                if obj['colour'] not in self.color_name_to_rgba:
                    rgba = self.color_name_to_rgba['white']
                else:
                    rgba = self.color_name_to_rgba[obj['colour']]
                utils.add_material(mat_name, Color=rgba)
            else:
                if obj['colour'] != self.object_properties[obj_fname]['color1']:
                    utils.add_color(self.color_name_to_rgba[obj['colour']])

    def _append_obj_to_scene(self, name, scale, position, orientation):
        # First figure out how many of this object are already in the scene so we can
        # give the new object a unique name
        count = 0
        for obj in bpy.data.objects:
            if obj.name.startswith(name):
                count += 1

        filename = os.path.join(self.shape_dir, '%s.blend' % name, 'Object', name)
        bpy.ops.wm.append(filename=filename)

        # Give it a new name to avoid conflicts
        new_name = '%s_%d' % (name, count)
        bpy.data.objects[name].name = new_name

        bpy.context.view_layer.objects.active = bpy.data.objects[new_name]
        bpy.data.objects[new_name].scale = (scale, scale, scale)
        bpy.data.objects[new_name].rotation_mode = 'QUATERNION'
        bpy.data.objects[new_name].rotation_quaternion = orientation
        bpy.data.objects[new_name].location = position

    def update_scene(self, objects, robot, gripper):
        obj_values = list(objects.values())
        for i, o in enumerate(self.objects):
            # pass
            o.rotation_mode = 'QUATERNION'
            o.location = obj_values[i][0]
            o.rotation_quaternion = obj_values[i][1]

        for k, v in robot.items():
            bpy.data.objects[k].rotation_mode = 'QUATERNION'
            bpy.data.objects[k].location = v[0]
            bpy.data.objects[k].rotation_quaternion = v[1]

        for k, v in gripper.items():
            bpy.data.objects[k].rotation_mode = 'QUATERNION'
            bpy.data.objects[k].location = v[0]
            bpy.data.objects[k].rotation_quaternion = v[1]

    def render(self, filename, segmentation=False):
        bpy.context.scene.render.filepath = filename
        while True:
            try:
                bpy.ops.render.render(write_still=True)
                break
            except Exception as e:
                print(e)

    def get_segmentation_masks(self):
        # Render shadeless and return list of colours
        f, path = tempfile.mkstemp(suffix='.png')
        object_colors = self.render_shadeless(self.objects, path)
        masks = self.assign_masks(object_colors, path)
        return masks

    def render_shadeless(self, blender_objects, path='flat.png'):
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

    def assign_masks(self, colors, image_path):
        masks = {}
        obj_masks = []
        img = cv2.imread(image_path)
        img_rgb = img[:, :, [2, 1, 0]]
        # Read shadeless render and encode masks in COCO format
        for color in colors:
            mask = np.all(img_rgb == color, axis=-1)
            mask = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            mask['counts'] = str(mask['counts'], "utf-8")
            obj_masks.append(mask)
        masks['objects'] = obj_masks
        table_mask = np.all(img_rgb == (80, 80, 80), axis=-1)
        table_mask = mask_utils.encode(np.asfortranarray(table_mask.astype(np.uint8)))
        table_mask['counts'] = str(table_mask['counts'], "utf-8")
        masks['table'] = table_mask
        robot_mask = np.all(img_rgb == (127, 127, 127), axis=-1)
        robot_mask = mask_utils.encode(np.asfortranarray(robot_mask.astype(np.uint8)))
        robot_mask['counts'] = str(robot_mask['counts'], "utf-8")
        masks['robot'] = robot_mask

        return masks