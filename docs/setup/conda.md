---
layout: default
title: Conda installation
parent: Setup
nav_order: 2
---

# Conda installation

We provide commands and scripts applicable for full conda environment. If you use miniconda, some adjustments to scripts will probably be necessery.
```bash
cd conda
conda create -n muble python=3.7.3
conda activate muble
bash blender_install.sh
pip install robosuite
bash robosuite_elements.sh
pip install opencv-python
pip install pycocotools
cd ..
```
Note that `blender_install.sh` uses sudo to put Blender executable in the `/bin` directory. Without sudo access you will have to adapt the script to choose any directory you have access to for the Blender installation. This will results in necessity of using full path directing to blender executable instead just `blender` whenever a script using Blender is invoked.

Note, that first time you will run MuJoCo-py/robosuite, MuJoCo will have to compile. To get it out of the way, you can optionally run immediately:
```bash
python -c "import mujoco_py"
```

Further, any time you want to use the envrironment, the following is sufficient:
```bash
conda activate muble
```
