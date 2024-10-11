'''
Based on:
clevr-dataset-gen
https://github.com/facebookresearch/clevr-dataset-gen
'''

import sys
import os
import bpy
import bpy_extras
import json
import math
from mathutils.bvhtree import BVHTree
# from mathutils import Vector
# import copy
"""
Some utility functions for interacting with Blender
"""


def get_boundings_size(prop_json, object_dir, output_file):
    """
    Reads out bounding box sizes from blend files corresponding to objects
    """
    with open(prop_json, 'r') as f:
        properties = json.load(f)

    obj_boxes = dict()
    for obj_name in properties['shapes'].values():
        filename = os.path.join(object_dir, '%s.blend' % obj_name)
        bpy.ops.wm.open_mainfile(filepath=filename)
        w = bpy.data.objects[obj_name].dimensions[0]
        h = bpy.data.objects[obj_name].dimensions[1]
        obj_boxes[obj_name] = {'w': w, 'h': h}

    with open(output_file, 'w') as f:
        json.dump(obj_boxes, f)


def extract_args(input_argv=None):
    """
    Pull out command-line arguments after "--". Blender ignores command-line flags
    after --, so this lets us forward command line arguments from the blender
    invocation to our own script.
    """
    if input_argv is None:
        input_argv = sys.argv
    output_argv = []
    if '--' in input_argv:
        idx = input_argv.index('--')
        output_argv = input_argv[(idx + 1):]
    return output_argv


def parse_args(parser, argv=None):
    return parser.parse_args(extract_args(argv))


def delete_object(obj):
    """ Delete a specified blender object """
    for o in bpy.data.objects:
        o.select_set(state=False)
    obj.select_set(state=True)
    bpy.ops.object.delete()


def get_camera_coords(cam, pos):
    """
    For a specified point, get both the 3D coordinates and 2D pixel-space
    coordinates of the point from the perspective of the camera.

    Inputs:
    - cam: Camera object
    - pos: Vector giving 3D world-space position

    Returns a tuple of:
    - (px, py, pz): px and py give 2D image-space coordinates; pz gives depth
      in the range [-1, 1]
    """
    scene = bpy.context.scene
    x, y, z = bpy_extras.object_utils.world_to_camera_view(scene, cam, pos)
    scale = scene.render.resolution_percentage / 100.0
    w = int(scale * scene.render.resolution_x)
    h = int(scale * scene.render.resolution_y)
    px = int(round(x * w))
    py = int(round(h - y * h))
    return (px, py, z)


def set_render(obj, val):
    """ Sets whether object is to be rendered or not """
    obj.hide_render = not val


def add_object(object_dir, name, scale, loc, theta=0):
    """
    Load an object from a file. We assume that in the directory object_dir, there
    is a file named "$name.blend" which contains a single object named "$name"
    that has unit size and is centered at the origin.

    - scale: scalar giving the size that the object should be in the scene
    - loc: tuple (x, y) giving the coordinates on the ground plane where the
      object should be placed.
    """
    # First figure out how many of this object are already in the scene so we can
    # give the new object a unique name
    count = 0
    for obj in bpy.data.objects:
        if obj.name.startswith(name):
            count += 1

    filename = os.path.join(object_dir, '%s.blend' % name, 'Object', name)
    bpy.ops.wm.append(filename=filename)

    # Give it a new name to avoid conflicts
    new_name = '%s_%d' % (name, count)
    bpy.data.objects[name].name = new_name

    # Set the new object as active, then rotate, scale, and translate it
    x, y = loc
    bpy.context.view_layer.objects.active = bpy.data.objects[new_name]
    bpy.context.object.rotation_euler[2] = math.radians(theta)
    bpy.ops.transform.resize(value=(scale, scale, scale))

    # Try placing in z=0
    # bpy.ops.transform.translate(value=(x + 0.45, y, 0.6))
    bpy.ops.transform.translate(value=(x, y, 0.0))


def add_object_quaternion(object_dir, name, scale, position, orientation):
    """
    Load an object from a file. We assume that in the directory object_dir, there
    is a file named "$name.blend" which contains a single object named "$name"
    that has unit size and is centered at the origin.

    - scale: scalar giving the size that the object should be in the scene
    - loc: tuple (x, y) giving the coordinates on the ground plane where the
      object should be placed.
    """
    # First figure out how many of this object are already in the scene so we can
    # give the new object a unique name
    count = 0
    for obj in bpy.data.objects:
        if obj.name.startswith(name):
            count += 1

    filename = os.path.join(object_dir, '%s.blend' % name, 'Object', name)
    bpy.ops.wm.append(filename=filename)

    # Give it a new name to avoid conflicts
    new_name = '%s_%d' % (name, count)
    bpy.data.objects[name].name = new_name

    # Set the new object as active, then rotate, scale, and translate it
    bpy.context.view_layer.objects.active = bpy.data.objects[new_name]
    # bpy.ops.collection.objects_remove_active()
    # bpy.context.view_layer.objects.active = bpy.data.objects[new_name]
    bpy.data.objects[new_name].scale = (scale, scale, scale)
    bpy.data.objects[new_name].rotation_mode = 'QUATERNION'
    bpy.data.objects[new_name].rotation_quaternion = (
        orientation['w'],
        orientation['x'],
        orientation['y'],
        orientation['z']
    )
    bpy.data.objects[new_name].location = (
        position['x'],
        position['y'],
        position['z']
    )


def load_materials(material_dir):
    """
    Load materials from a directory. We assume that the directory contains .blend
    files with one material each. The file X.blend has a single NodeTree item named
    X; this NodeTree item must have a "Color" input that accepts an RGBA value.
    """
    for fn in os.listdir(material_dir):
        if not fn.endswith('.blend'):
            continue
        name = os.path.splitext(fn)[0]
        filepath = os.path.join(material_dir, fn, 'NodeTree', name)
        bpy.ops.wm.append(filename=filepath)


def add_material(name, **properties):
    """
    Create a new material and assign it to the active object. "name" should be the
    name of a material that has been previously loaded using load_materials.
    """
    # Figure out how many materials are already in the scene
    mat_count = len(bpy.data.materials)

    # Create a new material; it is not attached to anything and
    # it will be called "Material"
    bpy.ops.material.new()

    # Get a reference to the material we just created and rename it;
    # then the next time we make a new material it will still be called
    # "Material" and we will still be able to look it up by name
    mat = bpy.data.materials['Material']
    mat.name = 'Material_%d' % mat_count

    # Attach the new material to the active object
    # Make sure it doesn't already have materials
    obj = bpy.context.active_object

    for i in range(0, len(obj.data.materials)):
        if 'Changable' in obj.data.materials[i].name:
            obj.data.materials[i] = mat

    # Find the output node of the new material
    output_node = None
    for n in mat.node_tree.nodes:
        if n.name == 'Material Output':
            output_node = n
            break

    # Add a new GroupNode to the node tree of the active material,
    # and copy the node tree from the preloaded node group to the
    # new group node. This copying seems to happen by-value, so
    # we can create multiple materials of the same type without them
    # clobbering each other
    group_node = mat.node_tree.nodes.new('ShaderNodeGroup')
    group_node.node_tree = bpy.data.node_groups[name]

    # Find and set the "Color" input of the new group node
    for inp in group_node.inputs:
        if inp.name in properties:
            inp.default_value = properties[inp.name]

    # Wire the output of the new group node to the input of
    # the MaterialOutput node
    mat.node_tree.links.new(
        group_node.outputs['Shader'],
        output_node.inputs['Surface'],
    )

    disp = False
    for o in group_node.outputs:
        if o.name == "Displacement":
            disp = True

    if disp:
        mat.node_tree.links.new(
            group_node.outputs['Displacement'],
            output_node.inputs['Displacement'],
        )


def add_ground(name):
    """
    Add ground material to the base
    """

    # Create a new material; it is not attached to anything and
    # it will be called "Material"
    bpy.ops.material.new()

    # Get a reference to the material we just created and rename it;
    # then the next time we make a new material it will still be called
    # "Material" and we will still be able to look it up by name
    mat = bpy.data.materials['Material']
    mat.name = 'Material_ground'

    obj = bpy.data.objects['Ground']
    # Attach the new material to the active object
    # Make sure it doesn't already have materials
    assert len(obj.data.materials) == 0
    obj.data.materials.append(mat)

    # Find the output node of the new material
    output_node = None
    for n in mat.node_tree.nodes:
        if n.name == 'Material Output':
            output_node = n
            break

    # Add a new GroupNode to the node tree of the active material,
    # and copy the node tree from the preloaded node group to the
    # new group node. This copying seems to happen by-value, so
    # we can create multiple materials of the same type without them
    # clobbering each other
    group_node = mat.node_tree.nodes.new('ShaderNodeGroup')
    group_node.node_tree = bpy.data.node_groups[name]

    # Wire the output of the new group node to the input of
    # the MaterialOutput node
    mat.node_tree.links.new(
        group_node.outputs['Shader'],
        output_node.inputs['Surface'],
    )

    disp = False
    for o in group_node.outputs:
        if o.name == "Displacement":
            disp = True

    if disp:
        mat.node_tree.links.new(
            group_node.outputs['Displacement'],
            output_node.inputs['Displacement'],
        )


def add_color(rgba):
    """
    Adds given colour to the object (if allowed)
    """
    obj = bpy.context.active_object

    for i in range(0, len(obj.data.materials)):
        if 'Changable' in obj.data.materials[i].name:
            mat = obj.data.materials[i]

    group_node = mat.node_tree.nodes['Group']
    for inp in group_node.inputs:
        if inp.name == 'Color':
            inp.default_value = rgba


def check_intersection_list(centres_list, size_list, margin):
    """
    Checks whether bounding boxes of objects intersect each other
    """

    # Check only pairs created by list[:-1] and last object
    # (as objects are added one by one)
    boxes_list = []
    for obj, siz in zip(centres_list, size_list):
        x_center = obj[0]
        y_center = obj[1]
        scale = obj[2]
        theta = math.radians(obj[3])
        wh = scale * siz['w'] / 2 + margin / 2
        hh = scale * siz['h'] / 2 + margin / 2
        corners_local = [
            {'x': wh, 'y': hh},
            {'x': -wh, 'y': hh},
            {'x': -wh, 'y': -hh},
            {'x': wh, 'y': -hh}]

        corners_global = []
        for c in corners_local:
            x_glob = c['x'] * math.cos(theta) - c['y'] * math.sin(theta)
            x_glob += x_center
            y_glob = c['x'] * math.sin(theta) + c['y'] * math.cos(theta)
            y_glob += y_center
            corners_global.append({'x': x_glob, 'y': y_glob})

        boxes_list.append(corners_global)

    for box in boxes_list[:-1]:
        if check_intersection(box, boxes_list[-1]):
            print("Intersection!")
            return True

    return False


def get_bbox_coords(centre_x, centre_y, theta, w, h):
    corners = []
    corners_local = [
        [w / 2, h / 2],
        [-w / 2, h / 2],
        [-w / 2, -h / 2],
        [w / 2, -h / 2]
    ]
    theta = math.radians(theta)
    for c in corners_local:
        x_glob = c[0] * math.cos(theta) - c[1] * math.sin(theta)
        x_glob += centre_x
        y_glob = c[0] * math.sin(theta) + c[1] * math.cos(theta)
        y_glob += centre_y
        corners.append([x_glob, y_glob])

    return corners


def check_intersection(box1, box2):
    """ Check intersection between 2 bounding boxes """
    boxes = [box1, box2]
    for box in boxes:
        for i in range(len(box)):
            vertex1 = box[i]
            vertex2 = box[(i + 1) % len(box)]

            perp_line = {'x': vertex2['y'] - vertex1['y'], 'y': vertex1['x'] - vertex2['x']}
            # Project everything on perpendicular line

            minA, maxA = None, None
            for vert in box1:
                projected = perp_line['x'] * vert['x'] + perp_line['y'] * vert['y']
                if minA is None or projected < minA:
                    minA = projected
                if maxA is None or projected > maxA:
                    maxA = projected

            minB, maxB = None, None
            for vert in box2:
                projected = perp_line['x'] * vert['x'] + perp_line['y'] * vert['y']
                if minB is None or projected < minB:
                    minB = projected
                if maxB is None or projected > maxB:
                    maxB = projected

            # No overlap of projection = no overlap of polygons
            if maxA < minB or maxB < minA:
                return False

    return True


def check_intersections_bvh(bvh_list, obj, epsilon=0.02):
    # return False, "A"
    intersections = False
    bpy.context.view_layer.objects.active = obj
    # print(obj)
    # print(obj.scale)
    # print(obj.matrix_world)
    scale = [obj.scale[0], obj.scale[1], obj.scale[2]]
    # scale = obj.scale
    new_x = scale[0] * (obj.dimensions.x + epsilon) / obj.dimensions.x
    new_y = scale[1] * (obj.dimensions.y + epsilon) / obj.dimensions.y
    # obj.dimensions += Vector((epsilon, epsilon, 0.2))
    # bpy.context.object.dimensions.y += epsilon
    # obj.dimensions.z += epsilon
    obj.scale = [new_x, new_y, scale[2]]
    # print(obj.scale)
    bpy.context.view_layer.update()
    # bpy.ops.transform.resize(value=(epsilon, epsilon, epsilon))
    target_mat = obj.matrix_world
    # print(target_mat)
    target_vert = [target_mat @ v.co for v in obj.data.vertices]
    target_poly = [p.vertices for p in obj.data.polygons]
    target_bvh = BVHTree.FromPolygons(target_vert, target_poly)
    # bpy.ops.transform.resize(value=(1.0 / epsilon, 1.0 / epsilon, 1.0 / epsilon))
    # obj.dimensions.x -= epsilon
    # obj.dimensions.y -= epsilon
    # obj.dimensions.z -= epsilon
    obj.scale = scale
    bpy.context.view_layer.update()

    for bvh in bvh_list:
        if target_bvh.overlap(bvh):
            # print("\n\n\n\nASDSADSADSAD\n\n\n\n\n\n")
            print("Intersection detected !!!")
            intersections = True

    # return False, target_bvh
    return intersections, target_bvh


































