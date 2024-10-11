---
layout: default
title: Demo
nav_order: 3
---

# Demo

We prepared some demo scripts to present scene, instruction, and sequence generation. By default, a new folder `output` will be created to save demo results. A set of results generated with demo scripts is also available at `demo_output`.

## Scene generation
Scene generation demo can be run as:
```bash
bash demo_scene_generation.sh
```
This will run the generation of 5 scenes containing 4-5 random objects. Resulting images will be saved at `output/scene_generation/images`, and scenes at `output/scene_generation/scenes`.

## Instruction generation
Instruction generation demo can be run as:
```bash
bash demo_instruction_generation.sh
```
This will generate a set of instructions for scenes available in `demo/demo_input`. Instructions will be saved at `output/instruction_generation/NS_AP_demo_instruction.json`, along with images of scenes with instructions imposed on them in the same directory.

## Sequence generation
Sequence generation demo can be run with Blender rendering, or with default MuJoCo visual output. To run it with Blender as a renderer, please use:
```bash
bash demo_sequence_generation.sh blender
```
The execution will prompt for the choice of one of the 10 instructions to execute. Each instruction represents a different task for scenes in `demo/demo_input`. As a result a directory with a corresponding index will be created in `output/sequence_generation`. Directory will contain a sequence of rendered frames, and a `.json` file with collected observation corresponding to sequence of actions.

Alternatively, default MuJoCo visualisation can be used:
```bash
bash demo_sequence_generation.sh mujoco
```
This will display the whole sequence (not only frames corresponding to action effect) on the computer's display. It has to be run in the environment with an access to a display (for docker: `run_wdisplay.sh`).