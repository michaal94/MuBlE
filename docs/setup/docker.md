---
layout: default
title: Docker installation
parent: Setup
nav_order: 1
---

# Docker installation

We prepare Dockerfile that can be used to create environment suitable for MuBlE.
Please make sure you have docker installed and running, and capable of [using gpu](https://github.com/NVIDIA/nvidia-docker).

To build docker, run:
```bash
cd docker
bash build.sh
```
To run container:
```bash
bash run.sh
```
To run container with display passthrough:
```bash
bash run_wdisplay.sh
```