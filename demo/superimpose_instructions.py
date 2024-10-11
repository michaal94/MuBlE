import os
import json
import argparse
from PIL import Image, ImageFont, ImageDraw

parser = argparse.ArgumentParser()
# Config file from instruction generator
parser.add_argument(
    "--config", "-c",
    default=os.path.join("configs", "default.json"),
    required=True
)

def main(args):
    h_text = 20
    font_text = 18
    font = ImageFont.truetype('./font.ttf', font_text)
    with open(args.config, 'r') as f:
        config = json.load(f)
    instr_file = config['output_instruction_file']
    out_dir = os.path.split(config['output_instruction_file'])[0]
    image_dir = config['input_scene_file']
    image_dir = os.path.split(image_dir)[0]
    image_dir = os.path.join(image_dir, 'images')
    instr_dict = {}
    with open(instr_file, 'r') as f:
        instructions = json.load(f)['instructions']
    for instr in instructions:
        if instr['image_filename'] not in instr_dict:
            instr_dict[instr['image_filename']] = [instr['instruction']]
        else:
            instr_dict[instr['image_filename']].append(instr['instruction'])
    for img_name in instr_dict.keys():
        img_path = os.path.join(image_dir, img_name)
        img = Image.open(img_path)
        w, h = img.size
        text_list = instr_dict[img_name]
        new_img = Image.new(
            img.mode, (w, h + h_text * len(text_list)), (37, 42, 52)
        )
        new_img.paste(img, (0, 0))
        save_path = os.path.join(out_dir, img_name)
        draw = ImageDraw.Draw(new_img)
        for idx, text in enumerate(text_list):
            draw.text(
                (5, h + idx * h_text + 1), text, fill="white", font=font
            )
        new_img.save(save_path)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
