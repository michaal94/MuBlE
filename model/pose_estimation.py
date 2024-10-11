import abc

class PoseEstimationModel:
    def __init__(self, params={}) -> None:
        pass

    @abc.abstractmethod
    def get_pose(self, image, observation=None):
        '''
        a
        '''
        pass

class PoseGTLoader(PoseEstimationModel):
    def get_pose(self, image, observation=None):
        assert 'objects' in observation, "Provide obs with GT"
        poses = []
        bboxes = []
        for tup in observation['objects']:
            poses.append((tup[1], tup[2]))
            bboxes.append(tup[3])
        return poses, bboxes
