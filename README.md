# Cricket: Tracing Compilation for Spherized Robots

Cricket is a library to trace the forward kinematics of spherized robots (generated through, for example, [`foam`](github.com/CoMMALab/foam/)).
It is built on [Pinocchio](https://github.com/stack-of-tasks/pinocchio) for forward kinematics, [CppAD](https://github.com/coin-or/CppAD) for tracing execution, [CppADCodeGen](https://github.com/joaoleal/CppADCodeGen) for generating code, and [CGAL](https://www.cgal.org/) for computing the bounding sphere of spheres.
It was used to generate the collision checking kernels in [VAMP](https://github.com/kavrakiLab/vamp) and [pRRTC](https://github.com/CoMMALab/pRRTC).

## Compilation Instructions

See either the provided Dockerfile or follow the instructions for compilation in a Conda environment.

Set up the environment:
```bash
micromamba env create -f environment.yaml
micromamba activate cricket
```

Build cricket:
```bash
cmake -GNinja -Bbuild .
cmake --build build
```

Run the script.
```bash
./build/fkcc_gen resources/panda.json

# Optionally format the code
clang-format -i panda_fk.hh
```


## Configuration

The script uses input JSON files that define what robot to load and what template to generate.
Cricket uses [inja](https://github.com/pantor/inja) to template code generation.
The configuration file specifies:
- The name of the robot
- Path of the URDF and SRDF relative (or absolute) to the configuration file
- The end-effector to use for attachments
- Collision checking resolution
- Output template and sub-templates to use
- Output filename

An example for the Franka Panda is given below:
```json
{
    "name": "Panda",
    "urdf": "panda/panda_spherized.urdf",
    "srdf": "panda/panda.srdf",
    "end_effector": "panda_grasptarget",
    "resolution": 32,
    "template": "templates/fk_template.hh",
    "subtemplates": [{"name": "ccfk", "template": "templates/ccfk_template.hh"}],
    "output": "panda_fk.hh"
}
```

In addition to the specified input fields, the script provides the following output information usable in your templates:
- `n_q`: number of joint DoF.
- `n_spheres`: number of collision spheres.
- `bound_lower`: lower bound of joint ranges.
- `bound_range`: range between lower and upper bound of joint ranges.
- `measure`: total measure of robot joint space.
- `end_effector_index`: frame index of end-effector.
- `min_radius`: minimum sphere radius on robot.
- `max_radius`: maximum sphere radius on robot.
- `joint_names`: name of joint corresponding to each DoF.
- `link_names`: name of frame corresponding to index.
- `per_link_spheres`: for each frame, the indices of the spheres associated with that frame.
- `links_with_geometry`: indices of the frames that have collision geometry.
- `bounding_sphere_index`: mapping between frame index to bounding sphere index (where bounding spheres are at the end of the sphere buffer, after `n_spheres`).
- `end_effector_collisions`: which frames the end-effector can collide with.
- `allowed_link_pairs`: sphere indices that are allowed to collide with each other.
- `eefk_code`, `eefk_code_vars`, `eefk_code_output`: C-style code for computing end-effector pose in position and quaternion, the number of intermediate variables used, and the number of output variables.
- `spherefk_code`, `spherefk_code_vars`, `spherefk_code_output`: C-style code for computing position of all spheres, the number of intermediate variables used, and the number of output variables.
- `ccfk_code`, `ccfk_code_vars`, `ccfk_code_output`: C-style code for computing position of all spheres and bounding spheres for collision checking, the number of intermediate variables used, and the number of output variables.
- `ccfkee_code`, `ccfkee_code_vars`, `ccfkee_code_output`: C-style code for computing position of all spheres and bounding spheres for collision checking as well as the position and quaternion for the end-effector, the number of intermediate variables used, and the number of output variables.


## Using with VAMP

To generate the required code for VAMP, simply use the provided `fkcc_gen` script with your desired robot configuration, e.g., for the Franka Panda:
```bash
./build/fkcc_gen panda.json

# Optionally format the code
clang-format -i panda_fk.hh
```

Then, copy this code into the `src/impl/vamp/robots/` folder in VAMP, where the other `<robot>.hh` files are:
```bash
# Assuming VAMP is located at ${VAMP_DIR}
cp panda_fk.hh ${VAMP_DIR}/src/impl/vamp/robots/
```

Add your robot's name (the `name` field of the configuration JSON) to VAMP's `pyproject.toml`. For the Panda, this would be:
```
VAMP_ROBOT_MODULES="panda"
VAMP_ROBOT_STRUCTS="Panda"
```
This is a list that is separated by semicolons (`;`), you can add or remove whatever robots you want.

Recompile VAMP:
```bash
pip install -e ${VAMP_DIR}
```

Test your robot out with `scripts/random_dance.py --robot <robot>`. For the Panda, this would be:
```bash
python scripts/random_dance.py --robot panda
```
