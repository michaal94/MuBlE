import abc

class VisualRecognitionLoader:
    def __init__(self, params={}) -> None:
        pass

    @abc.abstractmethod
    def get_scene(self, image, segmentation, scene_gt):
        '''
        Abstract class for the implementation of instruction -> symbolic program inference
        Expected input: dict: {'instruction': [...], ...}
        Expected output: list: [symbolic_function1, ...] 
        '''
        pass

class VisualGTLoader(VisualRecognitionLoader):
    def get_scene(self, image, segmentation, scene_gt):
        assert 'objects' in scene_gt, "Provide GT scene"
        return scene_gt['objects']
