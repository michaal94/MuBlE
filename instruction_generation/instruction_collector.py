import argparse
import json
import os
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


def collect_instructions(file_list, output_file, header):
    instructions = []
    split = None
    curr_len = 0
    for f_path in file_list:
        with open(f_path, 'r') as f:
            i_struct = json.load(f)
        if split is not None:
            msg = 'Input directory contains scenes from multiple splits'
            assert i_struct['instructions'][0]['split'] == split, msg
        else:
            split = i_struct['instructions'][0]['split']

        file_instructions = i_struct['instructions']
        file_instructions.sort(key=lambda x: x['instruction_idx'])
        for instr in file_instructions:
            instr['instruction_idx'] += curr_len
            print('Instruction number: ', str(instr['instruction_idx']))

        instructions += file_instructions
        curr_len = len(instructions)

    if split not in header:
        header['split'] = split

    output = {
        'info': header,
        'instructions': instructions
    }

    with open(output_file, 'w') as f:
        json.dump(output, f)


def main(args):
    input_files = sorted(os.listdir(args.input_dir))
    file_list = []
    for f_name in input_files:
        if not f_name.endswith('.json'):
            continue
        path = os.path.join(args.input_dir, f_name)
        file_list.append(path)
    header = {
        'version': args.version,
        'author': args.author,
        'date': args.date,
        'license': args.license
    }
    collect_instructions(file_list, args.output_file, header)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
