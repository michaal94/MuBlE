---
layout: default
title: Structure
parent: Dataset
nav_order: 2
---

# Structure

After unzipping (no matter what version of the dataset you have downloaded) you will see the directory `NS_AP_v1_0_a`. Inside this directory, the examples are split into separate tasks (for full, and reduced dataset) or a single task (for task separated files).

A structure of the catalogue, given by one example, looks as following:
```
NS_AP_v1_0_a
| -- move_multi
|    | -- test
|    |    | -- images
|    |    |    | -- NS_AP_train_000400.png
|    |    |    ` -- ...
|    |    | -- scenes
|    |    |    | -- NS_AP_train_000400.json
|    |    |    ` -- ...
|    |    | -- sequences
|    |    |    | -- NS_AP_train_000400
|    |    |    |    | -- frame_0000.png
|    |    |    |    | -- ...
|    |    |    |    ` -- sequence.json
|    |    |    ` -- ...
|    |    ` -- instructions.json
|    ` -- train
|      -- val
` -- move_single
  -- move_weight
  -- pick_up_lightest
  -- stack
  -- stack_three
  -- stack_weight
  -- weight_multi
  -- weight_order
  -- weight_single
```

Each task has its separate folder with training, validation and test directories inside. Further, each of those contain `images` directory containing images generated for the initial state of the scene. Next, `scenes` directory contains `.json` files with scene graphs corresponding to the generated scenes. A file `instructions.json` contains a structure with instructions corresponding to the scenes in the present folder. Finally, a `sequences` directory contains folders for every example in that task and split. Those folders contain `sequence.json` with structure of observations collected while generating the example, along with rendered scenes in the form `frame_XXXX.png`.

Note that all imgaes, scenes, sequences share the same prefix, also indexing is preserved throughout whole split (e.g. `move_multi` contains indexes from 4000 to 4999 for training and 400 to 499 for validation and test).