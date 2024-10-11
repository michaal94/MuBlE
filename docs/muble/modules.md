---
layout: default
title: Modules
parent: MuBlE
nav_order: 4
---

# Modules

We consider MuBlE as a modular framework. Therefore, we prepared an idea of the inference pipeline, located in `model.inference.InferenceTool`. A tool suggests capability of using the following modules:
- Visual Recognition - predicts visual properties of the objects
- Instruction Translation - translates textual instruction to symbolic programs
- Pose Prediction - predicts poses of all the objects in the scene
- Action Planner - given the goal task, returns primitive action to be currently undertaken

Note that for all the modules, a parent class is prepared, and a ground truth loading version can be invoked. In such way ground truth sequences may be generated (an example in `inference.py` in main directory).

Note also that modules can be adapted or removed, e.g. Action Executor can be removed, and you can use direct control instead of primitive actions.