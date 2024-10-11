---
layout: default
title: Adding new objects
parent: Environment
grand_parent: MuBlE
nav_order: 1
---

# Adding new objects

In order to add new objects to the MuJoCo simulation, we assume, that you have already prepared `.blend` files for Blener rendering. 

## Meshes
You have to generate convex meshes for simulation in MuJoCo. For concave objects, they have to be decomposed into a set of convex meshes. We suggest using a Blender [V-HACD add-on](https://github.com/andyp123/blender_vhacd) and saving the resulting convex hulls as a separate `.stl` files (using BatchMode in STL export dialogue window in Blender).

## XML files
Further, you have to create MuJoCo type assets, usually as an `.xml` file. You can use any file in `assets/train` as an example - meshes have to be loaded as assets, and placed as parts of the objects body.

## Path catalogure
To simplify pipeline we use `environment/path_catalogue.py` as a collection of paths to the assets. Additionally, if you decide to create new class there, remember about updating object loader class in `environment/objects.py`.