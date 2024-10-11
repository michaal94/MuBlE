import os
import json


class Config:
    def __init__(self):
        self.metadata = "metadata.json"
        self.template_dir = "templates"
        self.template_file = None
        self.synonyms = os.path.join("configs", "synonyms.json")
        self.plurals = os.path.join("configs", "plurals.json")
        self.substitutions = os.path.join("configs", "substitutions.json")

        self.input_scene_file = "scenes.json"
        self.output_instruction_file = os.path.join("../outputs",
                                                    "instructions.json")

        self.start_idx = 0
        self.num_scenes = -1
        # Prio on flattening task or template distribution
        self.task_distribution = True

        self.instructions_per_template = 1
        self.templates_per_scene = -1
        self.template_timeout = 60
        self.reset_statistics = 250
        self.extra_steps = 1

        self.verbose = False
        self.timer = False

        self.dump_freq = -1

    def merge_from_json(self, json_path):
        with open(json_path, 'r') as f:
            json_config = json.load(f)
        self.merge_from_dict(json_config)

    def merge_from_cmd(self, args_list):
        args_dict = dict()
        for i in range(len(args_list)):
            arg = args_list[i]
            if "--" in arg:
                key = arg.strip("-")
                if (i + 1) < len(args_list) and not ("--" in args_list[i + 1]):
                    value = args_list[i + 1]
                else:
                    value = True
                args_dict[key] = value
        self.merge_from_dict(args_dict)

    def merge_from_dict(self, args_dict):
        if "metadata" in args_dict:
            self.metadata = args_dict["metadata"]
        if "template_dir" in args_dict:
            self.template_dir = args_dict["template_dir"]
        if "template_file" in args_dict:
            self.template_file = args_dict["template_file"]
        if "instructions_per_template" in args_dict:
            self.instructions_per_template = args_dict[
                "instructions_per_template"
            ]
        if "templates_per_scene" in args_dict:
            self.templates_per_scene = args_dict["templates_per_scene"]
        if "synonyms" in args_dict:
            self.synonyms = args_dict["synonyms"]
        if "plurals" in args_dict:
            self.plurals = args_dict["plurals"]
        if "substitutions" in args_dict:
            self.substitutions = args_dict["substitutions"]
        if "output_instruction_file" in args_dict:
            self.output_instruction_file = args_dict["output_instruction_file"]
        if "start_idx" in args_dict:
            self.start_idx = args_dict["start_idx"]
        if "num_scenes" in args_dict:
            self.num_scenes = args_dict["num_scenes"]
        if "template_timeout" in args_dict:
            self.template_timeout = args_dict["template_timeout"]
        if "verbose" in args_dict:
            self.verbose = args_dict["verbose"]
        if "dump_freq" in args_dict:
            self.dump_freq = args_dict["dump_freq"]
        if "reset_statistics" in args_dict:
            self.reset_statistics = args_dict["reset_statistics"]
        if "task_distribution" in args_dict:
            self.task_distribution = args_dict["task_distribution"]
        if "timer" in args_dict:
            self.timer = args_dict["timer"]
        if "extra_steps" in args_dict:
            self.extra_steps = args_dict["extra_steps"]
        if "input_scene_file" in args_dict:
            self.input_scene_file = args_dict["input_scene_file"]

    def __str__(self):
        msg = ""
        for k, v in vars(self).items():
            msg += "{0:30}\t{1}\n".format(k, v)
        return msg
