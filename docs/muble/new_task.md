---
layout: default
title: New tasks
parent: MuBlE
nav_order: 5
---

Adding new tasks to MuBlE is simple. In order to do that you have to create your instructions with structuring the ground truth program to include your new subgoals.

E.g. when creating task that requires measuring the stiffness of the material, you have to add something like `measure_stiffness` to the program (see template in [Intruction generation](/docs/muble/instruction_gen.html)).

Further, to enable automatic ground truth generation, you can modify ActionPlanner class to respond to new programs.