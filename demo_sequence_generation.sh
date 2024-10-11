#!/bin/bash
if [ $1 == "blender" ]; then
    echo "Running demo in Blender"
    blender -b -noaudio --python demo_sequence_generation.py -- --input_instruction_json ./demo/demo_input/NS_AP_demo_instructions.json --input_scene_dir ./demo/demo_input/scenes --output_dir ./output/sequence_generation
elif [ $1 == "mujoco" ]; then
    echo "Running with MuJoCo display"
    python demo_sequence_generation.py --input_instruction_json ./demo/demo_input/NS_AP_demo_instructions.json --input_scene_dir ./demo/demo_input/scenes --output_dir ./output/sequence_generation
else
    echo "Provide an argument where to display output: [blender, mujoco]"
fi