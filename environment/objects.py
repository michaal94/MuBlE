from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import array_to_string, find_elements, xml_path_completion
from environment.path_catalogue import ShopVrbXMLFiles
import os

class ShopVrbObject(MujocoXMLObject):
    def __init__(self, name, file, scale=1.0, mode='train'):
        xml_path = ShopVrbXMLFiles.get(file, mode)
        
        with open(xml_path, 'r') as f:
            xml_data = f.read()

        xml_data = xml_data.replace(
            'scale="1 1 1"',
            f'scale="{scale} {scale} {scale}"',
        )
        temp_file = os.path.dirname(xml_path)
        temp_file = os.path.join(temp_file, 'temp_obj.xml')
        with open(temp_file, 'w') as f:
            f.write(xml_data)

        super().__init__(
            # xml_path,
            temp_file,
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )

class CoordinateVis(MujocoXMLObject):
    def __init__(self, name):
        xml_path = './assets/coord_vis.xml'

        super().__init__(
            xml_path,
            name=name,
            joints=None,
            obj_type="visual",
            duplicate_collision_geoms=True,
        )