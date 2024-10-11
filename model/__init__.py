from .instruction_inference import InstructionGTLoader
from .visual_recognition import VisualGTLoader
from .action_planner import ActionGTPlanner
from .pose_estimation import PoseGTLoader

INSTRUCTION_MODELS = {
    "GTLoader": InstructionGTLoader
}

VISUAL_RECOGNITION_MODELS = {
    "GTLoader": VisualGTLoader
}

ACTION_PLAN_MODELS = {
    "GTLoader": ActionGTPlanner
}

POSE_MODELS = {
    "GTLoader": PoseGTLoader
}