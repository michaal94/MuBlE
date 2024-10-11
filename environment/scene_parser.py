import numpy as np

class SceneParser():
    def __init__(self) -> None:
        # For now nothing, maybe some config later
        pass

    def parse_scene(self, scene_struct):
        obj_name_to_attr = {}
        obj_counter = {}
        for obj in scene_struct['objects']:
            file_name = obj['file']
            if file_name not in obj_counter:
                obj_counter[file_name] = 0
            else:
                obj_counter[file_name] += 1
            obj_id = f'{file_name}_{obj_counter[file_name]:01d}'
            # print(obj_id)
            obj_name_to_attr[obj_id] = {
                'obj_name': obj['name'],
                'file_name': obj['file'],
                'position': obj['3d_coords'],
                'orientation': obj['orientation'],
                'scale': obj['scale_factor']
            }
            if 'bbox' in obj:
                obj_name_to_attr[obj_id]['bbox'] = self._get_local_bounding_box(
                    obj['bbox']
                )

        return obj_name_to_attr

    def _get_local_bounding_box(self, bbox):
        bbox_wh = bbox['x'] / 2
        bbox_dh = bbox['y'] / 2
        bbox_h = bbox['z']

        # Local bounding box w.r.t to local (0,0,0) - mid bottom
        return np.array(
            [
                [ bbox_wh,  bbox_dh, 0],
                [-bbox_wh,  bbox_dh, 0],
                [-bbox_wh, -bbox_dh, 0],
                [ bbox_wh, -bbox_dh, 0],
                [ bbox_wh,  bbox_dh, bbox_h],
                [-bbox_wh,  bbox_dh, bbox_h],
                [-bbox_wh, -bbox_dh, bbox_h],
                [ bbox_wh, -bbox_dh, bbox_h]
            ]
        )