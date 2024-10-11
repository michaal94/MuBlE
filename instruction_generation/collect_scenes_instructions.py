import argparse
from dis import Instruction
import json
import os
from datetime import datetime
import copy
import re
import shutil
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_dir_images", '-iimg',
    required=True
)
parser.add_argument(
    "--input_dir_scenes", '-iscen',
    required=True
)
parser.add_argument(
    "--input_instruction", '-iinstr',
    required=True
)
parser.add_argument(
    "--output_dir", '-o',
    required=True,
    default=os.path.join("../outputs_refined")
)
parser.add_argument(
    "--indent", default=False, action="store_true"
)

def main(args):
    new_to_old = {}
    numbering = 0
    with open(args.input_instruction, 'r') as f:
        instruction_struct = json.load(f)
    new_instruction_struct = {}
    new_instruction_struct["info"] = instruction_struct["info"]
    new_instruction_struct["instructions"] = []

    # image_files = sorted(os.listdir(args.input_dir_images))
    # scene_files = sorted(os.listdir(args.input_dir_scenes))

    img_dir = os.path.basename(os.path.normpath(args.input_dir_images))
    scene_dir = os.path.basename(os.path.normpath(args.input_dir_scenes))
    instr_name = os.path.basename(os.path.normpath(args.input_instruction))

    out_img_dir = os.path.join(args.output_dir, img_dir)
    out_scene_dir = os.path.join(args.output_dir, scene_dir)

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_scene_dir, exist_ok=True)

    for instruction in tqdm(instruction_struct["instructions"]):
        img_fname = instruction['image_filename']
        name_ext = os.path.splitext(img_fname)
        root_name = name_ext[0]
        img_ext = name_ext[1]
        scene_fname = f"{root_name}.json"
        img_path = os.path.join(args.input_dir_images, img_fname)
        scene_path = os.path.join(args.input_dir_scenes, scene_fname)

        number = re.search(r"\d+$", root_name).group()
        num_len = len(number)
        new_number = str(numbering).zfill(num_len)

        new_root = root_name.replace(number, new_number)
        new_to_old[new_root] = root_name
        new_img_path = os.path.join(out_img_dir, new_root + img_ext)
        new_scene_path = os.path.join(out_scene_dir, new_root + '.json')
        instruction["image_filename"] = new_root + img_ext
        instruction["image_index"] = numbering
        new_instruction_struct['instructions'].append(instruction)
        
        with open(scene_path, 'r') as f:
            scene_struct = json.load(f)

        scene_struct["image_filename"] = new_root + img_ext
        scene_struct["image_index"] = numbering

        with open(new_scene_path, 'w') as f:
            json.dump(scene_struct, f, indent=4)
            
        shutil.copyfile(img_path, new_img_path)

        numbering += 1

    with open(os.path.join(args.output_dir, instr_name), 'w') as f:
        json.dump(new_instruction_struct, f, indent=4)
    
    with open(os.path.join(args.output_dir, 'new_to_old.json'), 'w') as f:
        json.dump(new_to_old, f, indent=4)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
