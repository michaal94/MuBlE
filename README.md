# MuBlE: MuJoCo and Blender simulation Environment for task planning in Robotics Manipulation

## Usage

To use the framework, run demo scripts, please follow the [documentation](https://michaal94.github.io/MuBlE)

## Export to robotsim

`robotsim_export.py` writes MuBlE scenes into the neutral handoff format read by
[robotsim](https://github.com/crustos/robotsim), a Blender-based simulator that
renders aligned photorealistic, line-art, object-index and metric-depth passes
from one viewpoint, and executes real control firmware against the simulated
plant.

The two projects make opposite trades — MuBlE spends its effort on contact
fidelity and task structure, robotsim on the render modalities — so the handoff
is more useful than a merge.

```sh
./robotsim_export.py demo_output/scene_generaion/NS_AP_scenes.json -o out/
./robotsim_export.py scenes.json --index 3 -o scene3.json
```

Each object carries its shape name, scale, orientation quaternion, MuBlE's own
`3d_coords` origin, the convex-hull decomposition from `meshes/`, and whether
its appearance was randomised for that scene. robotsim appends the authored
`.blend` from `scene_generation/data/shapes` for rendering and loads the hulls
into MuJoCo for collision, so both the image and the physics are MuBlE's rather
than an approximation.

Nothing in `environment` is imported at module scope, so exporting a generated
scene file needs neither robosuite, MuJoCo nor Blender installed.

### Notes

- Object ids start at 1; 0 is background in robotsim's object-index pass.
- Only material slots named `Changable` are overridden, matching
  `scene_generation/utils.add_material`. The documentation in
  `docs/muble/scene_gen.md` gives this as *Changeable*; the code does not, and
  the code is what runs. Matching the documented spelling silently repaints
  nothing.
- `3d_coords` is an object's mid-bottom, not its centre. The exporter emits both
  that (`origin`, for the real mesh) and the bounding-box centre (`position`,
  for consumers that only have the box).
