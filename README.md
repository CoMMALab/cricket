# Cricket: Tracing Compilation for Spherized Robots

Cricket is a library to trace the forward kinematics of spherized robots (generated through, for example, [`foam`](github.com/CoMMALab/foam/)).
It is built on [Pinocchio](https://github.com/stack-of-tasks/pinocchio) for forward kinematics, [CppAD](https://github.com/coin-or/CppAD) for tracing execution, [CppADCodeGen](https://github.com/joaoleal/CppADCodeGen) for generating code, and [CGAL](https://www.cgal.org/) for computing the bounding sphere of spheres.
It was used to generate the collision checking kernels in [VAMP](https://github.com/kavrakiLab/vamp) and [pRRTC](https://github.com/CoMMALab/pRRTC).

## Compilation Instructions

Follow the instructions for compilation in a Conda environment.

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
