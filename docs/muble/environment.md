---
layout: default
title: Environment
parent: MuBlE
nav_order: 3
has_children: true
---

# Environment

We provide a short overview of the most important classes in the environment. They can be modified and adjusted according to the neeeds.

## TabletopEnv

Located in `environment.tabletop_env.TabletopEnv`, TabletopEnv contains the main simulation mechanisms. It inherits after [robosuite](https://github.com/ARISE-Initiative/robosuite) class `SingleArmEnv`. See the class itself for its parameters.

### Sensors
In the `_setup_observables()` function you can affect the observations returned from the simulation by coding new sensors.

### Blender rendering
Function `blender_render()` provides functionality for rendering in Blender (if having had it loaded beforehand). 

### Controllers
You can use default [robosuite](https://github.com/ARISE-Initiative/robosuite) controllers. However, we noted some problems with the stability of OSC controller, and we provided slightly tweaked implementation in `environment.controller.CustomOperationalSpaceController`

## ActionExecutor
Located in `environment.actions.ActionExecutor`, ActionExecutor provides implementation for primitive actions:
- move
- approach
- open_gripper
- close_gripper
- lift
- lower 

Calling `.step()` on the instance of action executor provides a steering vector leading to the execution of the current action.

## BlenderRenderer
Located in `environment.renderer.BlenderRenderer`, BlenderRenderer provides implementation for the use of Blender rendering with MuJoCo simulation. It main tasks are to `.update_scene()`, `.render()`, `.get_segmentation_masks()`.

## SceneParser
Located in `environment.scene_parser.SceneParser`, SceneParser provides tool for translating the scene from `.json` obtained via scene generator to internal framework format.

## ProgramExecutor
Located in `model.program_executor.ProgramExecutor`, ProgramExecutor provides interface of executing [CLEVR-IEP](https://arxiv.org/abs/1705.03633) style programs on the given scene. We extended the capability of the program execution to return either answer (if possible) or task. Task can be defined as a final task in template instruction, or when ProgramExecutor encounters unknown property, it triggers the task `measure_property`.
