#!/usr/bin/env python3
"""
Export MuBlE scenes into robotsim.

MuBlE and robotsim make opposite trades. MuBlE couples MuJoCo physics to Blender
rendering for long-horizon manipulation, and puts its effort into contact
fidelity and task structure. robotsim keeps contact kinematic and puts its effort
into the render modalities -- photorealistic, line art, object index and metric
depth from one viewpoint -- and into executing real control firmware. Neither
subsumes the other, which is why a handoff is more useful than a merge.

This writes the neutral description robotsim reads: object poses, box extents and
stable integer ids, plus the camera. robotsim's `muble_bridge` turns that into
physics geometry, render proxies, or both.

The point of the integer ids is the semantic map. robotsim supervises its
segmentation from Blender's object-index pass, so an id assigned here survives
all the way to the label a perception network is trained against. Ids are
therefore assigned once, written down, and never recomputed from a sort order
that a later scene could change.

Usage:

    ./robotsim_export.py demo_output/scene_generaion/NS_AP_scenes.json -o out/
    ./robotsim_export.py scenes.json --index 3 -o scene3.json

Deliberately imports nothing from `environment` at module scope, so exporting a
generated scene file does not require robosuite, MuJoCo or Blender to be
installed. `from_env()` is the live-environment path and imports lazily.
"""

import argparse
import json
import math
import os
import sys


FORMAT = 'robotsim/muble-handoff'
## Bumped when the handoff gains fields robotsim relies on. v2 added the asset
## references that make the RGB pass match MuBlE's own render rather than
## approximating it with boxes.
VERSION = 2


def yaw_to_quat(yaw):
    """Yaw about +Z as a (w, x, y, z) quaternion."""
    return (math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5))


def quat_to_yaw(quat):
    """
    Heading from a (w, x, y, z) quaternion.

    MuBlE stores orientation as a quaternion and *also* as a `rotation` angle in
    degrees. They agree, but the quaternion is the one the simulator uses, so it
    is the one trusted here.
    """
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def object_box(obj):
    """
    World-space centre and full extent of one MuBlE object.

    MuBlE's `3d_coords` is the mid-*bottom* of the object, not its centre -- the
    local bounding box in `scene_parser` is built from (0, 0, 0) at the base.
    Treating it as a centre sinks every object half its own height into the
    table, which looks like a physics bug and is not one.
    """
    bbox = obj.get('bbox')
    if not bbox:
        return None
    x, y, z = obj['3d_coords']
    ex, ey, ez = bbox['x'], bbox['y'], bbox['z']
    return (x, y, z + ez * 0.5), (ex, ey, ez)


## Where MuBlE keeps the things robotsim needs, relative to the repository root.
## Recorded in the handoff as relative paths plus one absolute root, so a corpus
## generated on another machine only has to be told where MuBlE was checked out.
ASSET_DIRS = {
    'shapes': 'scene_generation/data/shapes',
    'materials': 'scene_generation/data/materials',
    'meshes': 'meshes/train',
}

## MuBlE's own default property tables. Read to work out whether an object's
## appearance was overridden for this scene or is baked into its .blend.
PROPERTIES = 'scene_generation/data/properties_ns_ap.json'
OBJECT_PROPERTIES = 'scene_generation/data/object_properties_ns_ap.json'


def repo_root():
    return os.path.dirname(os.path.abspath(__file__))


def load_properties(root=None, properties=None, object_properties=None):
    """
    The two tables that decide an object's appearance.

    Returned as (properties, object_properties), either possibly empty: a
    missing table means "assume nothing was overridden", which is the right
    default because most MuBlE objects carry their appearance in their .blend.
    """
    root = root or repo_root()
    out = []
    for path, default in ((properties, PROPERTIES),
                          (object_properties, OBJECT_PROPERTIES)):
        path = path or os.path.join(root, default)
        try:
            with open(path) as handle:
                out.append(json.load(handle))
        except (OSError, ValueError):
            out.append({})
    return out[0], out[1]


def material_override(shape, scene_obj, properties, object_properties):
    """
    The material robotsim must apply, or None to use the .blend's own.

    This is the part that decides whether the RGB pass matches MuBlE's render.
    MuBlE bakes each object's appearance into its .blend and only overrides the
    material slots literally named 'Changable' -- note the spelling, which the
    documentation gives as 'Changeable' and the code does not. An object whose
    properties say nothing changed must be left exactly as it was authored;
    repainting it is how a mug that should be black comes out default grey.
    """
    props = object_properties.get(shape)
    if not props:
        return None

    override = {}
    if props.get('change_material') and scene_obj.get('material'):
        ## 'metal' -> the 'Metal' NodeTree appended from materials/Metal.blend.
        node = properties.get('materials', {}).get(scene_obj['material'])
        if node:
            override['node_tree'] = node
    if props.get('change_color1') and scene_obj.get('colour'):
        rgb = properties.get('colors', {}).get(scene_obj['colour'])
        if rgb:
            ## MuBlE stores 0-255; Blender wants 0-1 RGBA.
            override['color'] = [c / 255.0 for c in rgb] + [1.0]
    return override or None


def collision_hulls(shape, root=None, meshes=None):
    """
    The convex decomposition MuBlE ships for this object, in file order.

    Sorted numerically rather than lexically: hull_10 sorts before hull_2 as a
    string, and while the order does not change the union of the hulls, a stable
    order keeps two runs of the same scene producing byte-identical MJCF.
    """
    root = root or repo_root()
    directory = os.path.join(root, meshes or ASSET_DIRS['meshes'], shape)
    if not os.path.isdir(directory):
        return []
    names = [f for f in os.listdir(directory)
             if f.startswith('%s_hull_' % shape) and f.endswith('.stl')]

    def order(name):
        stem = name[len('%s_hull_' % shape):-len('.stl')]
        return int(stem) if stem.isdigit() else 0

    return [os.path.join(meshes or ASSET_DIRS['meshes'], shape, n)
            for n in sorted(names, key=order)]


def visual_mesh(shape, root=None, meshes=None):
    """The single render mesh, for consumers that cannot append a .blend."""
    root = root or repo_root()
    relative = os.path.join(meshes or ASSET_DIRS['meshes'], shape,
                            'simple_mesh.stl')
    return relative if os.path.isfile(os.path.join(root, relative)) else None


def export_scene(scene, index=0, table_size=(1.2, 1.8, 0.05), root=None,
                 properties=None, object_properties=None):
    """
    One MuBlE scene as a robotsim handoff dict.

    Object ids start at 1 because 0 is the background in an object-index pass,
    and a foreground object sharing the background's label is indistinguishable
    from empty space in the trained semantic map.

    Every object carries two descriptions of where it is, and they are not
    interchangeable. `origin` is MuBlE's own `3d_coords`, the object's
    mid-bottom, and is what the real mesh must be placed at -- MuBlE's
    `add_object_quaternion` assigns it to `location` directly. `position` is the
    centre of the bounding box, half an object higher, and is only for the box
    fallback. Using one where the other belongs buries every object half its own
    height in the table, or floats it.
    """
    root = root or repo_root()
    props, obj_props = load_properties(root, properties, object_properties)

    objects = []
    for i, obj in enumerate(scene.get('objects', [])):
        box = object_box(obj)
        if box is None:
            continue
        centre, extent = box
        shape = obj.get('file', 'object')
        quat = obj.get('orientation')
        if quat:
            quat = list(quat)
            yaw = quat_to_yaw(quat)
        else:
            yaw = math.radians(obj.get('rotation', 0.0))
            quat = list(yaw_to_quat(yaw))

        objects.append({
            'id': i + 1,
            ## Unique per scene: two mugs must not collide in the name table, or
            ## the semantic map silently labels one of them as the other.
            'name': '%s_%02d' % (shape, i),
            'label': obj.get('name', shape),
            'shape': shape,
            'origin': list(obj['3d_coords']),
            'position': list(centre),
            'size': list(extent),
            'scale': obj.get('scale_factor', 1.0),
            'quaternion': quat,
            'yaw': yaw,
            'movable': obj.get('movability') != 'fixed',
            'mass': (obj['weight_gt'] / 1000.0
                     if obj.get('weight_gt') is not None else None),
            'material': obj.get('material'),
            'colour': obj.get('colour'),
            'material_override': material_override(shape, obj, props, obj_props),
            'visual_mesh': visual_mesh(shape, root),
            'collision': collision_hulls(shape, root),
        })

    out = {
        'format': FORMAT,
        'version': VERSION,
        'index': scene.get('image_index', index),
        'source': scene.get('image_filename'),
        ## Relative paths plus one absolute root: the handoff stays portable and
        ## robotsim only needs telling where MuBlE lives if it has moved.
        'assets': dict(ASSET_DIRS, root=root),
        'objects': objects,
    }

    ## The table is the ground plane as far as a driving robot is concerned, and
    ## MuBlE records it only as a segmentation mask, so its extent is a
    ## parameter rather than something that can be read back out.
    if 'table' in scene:
        out['table'] = {
            'name': 'TABLE', 'id': 0,
            'position': [0.0, 0.0, -table_size[2] * 0.5],
            'size': list(table_size),
        }

    camera = scene.get('camera_params')
    if camera:
        out['camera'] = {
            'position': list(camera['position']),
            ## MuBlE writes Blender euler angles in degrees; robotsim works in
            ## radians throughout, so the conversion happens once, here.
            'rotation': [math.radians(a) for a in camera['rotation']],
        }
    return out


def load_scenes(path):
    """Accept either MuBlE's {'scenes': [...]} bundle or a bare list."""
    with open(path) as handle:
        data = json.load(handle)
    if isinstance(data, dict) and 'scenes' in data:
        return data['scenes']
    return data if isinstance(data, list) else [data]


def from_env(env, index=0):
    """
    Export a live TabletopEnv rather than a generated scene file.

    Imported lazily and guarded: the whole point of keeping `environment` out of
    this module's imports is that the file-based path works without robosuite.
    """
    try:
        import numpy as np
    except ImportError as exc:                                # pragma: no cover
        raise ImportError('from_env needs numpy') from exc

    objects = []
    for i, obj in enumerate(getattr(env, 'objects', [])):
        name = obj.name
        pos = np.array(env.get_object_position(name), dtype=float)
        quat = env.get_object_orientation(name)          ## xyzw from robosuite
        yaw = quat_to_yaw((quat[3], quat[0], quat[1], quat[2]))
        size = getattr(obj, 'bbox', None)
        objects.append({
            'id': i + 1,
            'name': '%s_%02d' % (name, i),
            'label': name,
            'position': pos.tolist(),
            'size': list(size) if size is not None else [0.1, 0.1, 0.1],
            'yaw': yaw,
            'movable': True,
            'mass': None,
        })
    return {'format': FORMAT, 'version': VERSION, 'index': index,
            'source': 'live-env', 'objects': objects}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('scenes', help='MuBlE scenes JSON')
    parser.add_argument('-o', '--out', default='robotsim_scenes',
                        help='output file, or directory when exporting all')
    parser.add_argument('--index', type=int, default=None,
                        help='export only this scene')
    parser.add_argument('--table-size', type=float, nargs=3,
                        default=(1.2, 1.8, 0.05), metavar=('X', 'Y', 'Z'))
    parser.add_argument('--root', default=None,
                        help='MuBlE checkout to resolve assets against '
                             '(default: this script\'s own repository)')
    args = parser.parse_args(argv)

    scenes = load_scenes(args.scenes)
    if args.index is not None:
        if not 0 <= args.index < len(scenes):
            parser.error('scene %d out of range (%d in file)'
                         % (args.index, len(scenes)))
        scenes = [scenes[args.index]]

    exported = [export_scene(s, i, tuple(args.table_size), root=args.root)
                for i, s in enumerate(scenes)]

    if len(exported) == 1 and not args.out.endswith(os.sep):
        path = args.out if args.out.endswith('.json') else args.out + '.json'
        with open(path, 'w') as handle:
            json.dump(exported[0], handle, indent=1)
        scene = exported[0]
        meshed = sum(1 for o in scene['objects'] if o['collision'])
        print('wrote %s (%d objects, %d with shipped geometry)'
              % (path, len(scene['objects']), meshed))
        return 0

    os.makedirs(args.out, exist_ok=True)
    for scene in exported:
        path = os.path.join(args.out, 'scene_%06d.json' % scene['index'])
        with open(path, 'w') as handle:
            json.dump(scene, handle, indent=1)
    print('wrote %d scenes to %s/' % (len(exported), args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
