'''
Based on:
clevr-dataset-gen
https://github.com/facebookresearch/clevr-dataset-gen
'''

import argparse
import json
import os
from datetime import datetime as dt

"""
During rendering, each SHOP-VRB scene file is dumped to disk as a separate JSON
file; this is convenient for distributing rendering across multiple machines.
This script collects all SHOP-VRB scene files stored in a directory and combines
them into a single JSON file. This script also adds the version number, date,
and license to the output file.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='../output/scenes')
parser.add_argument('--output_file', default='../output/SHOP_VRB_scenes.json')
parser.add_argument('--version', default='1.0')
parser.add_argument('--date', default=dt.today().strftime("%d/%m/%Y"))
parser.add_argument('--license', default='Creative Commons Attribution (CC-BY 4.0)')


def main(args):
    scenes = []
    split = None
    for filename in os.listdir(args.input_dir):
        if not filename.endswith('.json'):
            continue
        path = os.path.join(args.input_dir, filename)
        with open(path, 'r') as f:
            scene = json.load(f)
        scenes.append(scene)
        if split is not None:
            msg = 'Input directory contains scenes from multiple splits'
            assert scene['split'] == split, msg
        else:
            split = scene['split']
    scenes.sort(key=lambda s: s['image_index'])
    for s in scenes:
        print(s['image_filename'])
    output = {
        'info': {
            'date': args.date,
            'version': args.version,
            'split': split,
            'license': args.license,
        },
        'scenes': scenes
    }
    with open(args.output_file, 'w') as f:
        json.dump(output, f)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
