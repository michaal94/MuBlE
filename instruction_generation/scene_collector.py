import os
import json
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_dir", '-i',
    required=True
)
parser.add_argument(
    "--output_file", '-o',
    required=True,
    default=os.path.join("outputs", "scenes.json")
)
parser.add_argument(
    "--author",
    default="Michal Nazarczuk"
)
parser.add_argument(
    "--name", default="TODO name"
)
parser.add_argument(
    "--date", default=datetime.today().strftime("%d/%m/%Y")
)
parser.add_argument(
    "--license", default="CC-BY"
)
parser.add_argument(
    "--version", default="0.0.1a"
)
parser.add_argument(
    "--indent", default=False, action="store_true"
)


def main(args):
    scenes = []
    split = None
    for filename in sorted(os.listdir(args.input_dir)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(args.input_dir, filename)
        with open(path, 'r') as f:
            scene = json.load(f)
        if split is not None:
            msg = 'Input directory contains scenes from multiple splits'
            assert scene['split'] == split, msg
        else:
            split = scene['split']
        scenes.append(scene)
    scenes.sort(key=lambda s: s['scene_index'])
    for s in scenes:
        print(s['image_filename'])
    output = {
        'info': {
            'split': split,
            'version': args.version,
            'author': args.author,
            'date': args.date,
            'license': args.license
        },
        'scenes': scenes
    }
    with open(args.output_file, 'w') as f:
        if args.indent:
            json.dump(output, f, indent=4)
        else:
            json.dump(output, f)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
