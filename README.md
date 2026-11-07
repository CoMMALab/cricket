[![Docker Image CI](https://github.com/CoMMALab/cricket/actions/workflows/docker-build.yml/badge.svg)](https://github.com/CoMMALab/cricket/actions/workflows/docker-build.yml)

# Cricket: Tracing Compilation for Spherized Robots

Cricket is a library to trace the forward kinematics of spherized robots (generated through, for example, [`foam`](github.com/CoMMALab/foam/)).
It is built on [Pinocchio](https://github.com/stack-of-tasks/pinocchio) for forward kinematics, [CppAD](https://github.com/coin-or/CppAD) for tracing execution, [CppADCodeGen](https://github.com/joaoleal/CppADCodeGen) for generating code, and [CGAL](https://www.cgal.org/) for computing the bounding sphere of spheres.
It was used to generate the collision checking kernels in [VAMP](https://github.com/kavrakiLab/vamp) and [pRRTC](https://github.com/CoMMALab/pRRTC).

## Compilation Instructions

See either the provided Dockerfile or follow the instructions for compilation in a Conda environment.

### Conda/Mamba Installation

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

### Pip Install

Cricket also ships a Python package (the C++ library + Python bindings via [nanobind](https://github.com/wjakob/nanobind)). After setting up the conda env above:
```bash
pip install .
```
Or for a dev install:
```bash
pip install --no-build-isolation -e .
```
This installs `cricket` as an importable module and places `fkcc_gen`, headers, and CMake config files under `<env>/lib/python3.X/site-packages/cricket/`. 
CMake builds of downstream projects can locate it via `find_package(cricket CONFIG REQUIRED)`.

### Docker Installation

Build the container:
```bash
docker build . -t cricket
```

Run the script:
```bash
docker run --rm -v "$(pwd):/mount" --user "$(id -u):$(id -g)" cricket:latest /mount/resources/panda.json -t /mount/panda.hh

# Optionally format the code
clang-format -i panda_fk.hh
```
Note the use of the `/mount` directory and specification of output file.

## JIT Runtime

In addition to the `fkcc_gen` CLI, cricket optionally builds a small C++ library (`cricket::cricket_jit`) that wraps clang + LLVM's ORC JIT to compile and load generated robot code at runtime.
This is used by [VAMP](https://github.com/kavrakiLab/vamp)'s `vamp.jit.load_robot(...)` to specialize a planner for an arbitrary URDF without full rebuilding.

### Building

Building the JIT library is **on by default**. It requires LLVM + Clang development libraries and a `clang` executable on `PATH` at runtime.
To disable:
```bash
cmake -GNinja -Bbuild -DCRICKET_BUILD_JIT=OFF .
```

## Python API

The Python bindings expose the code generation pipeline as functions:

```python
import cricket

opts = cricket.GenOptions(
    urdf="path/to/panda_spherized.urdf",
    srdf="path/to/panda.srdf",
    end_effector="panda_grasptarget",
    template_path=cricket.resources_dir() / "templates" / "fk_template.hh",
    data={"name": "Panda", "resolution": 32},
)
gen = cricket.generate_robot_source(opts)
print(gen.robot_name, gen.dimension, gen.n_spheres)
```

## Configuration

The script uses input JSON files that define what robot to load and what template to generate.
Cricket uses [inja](https://github.com/pantor/inja) to template code generation.
The configuration file specifies:
- The name of the robot
- Path of the URDF and SRDF relative (or absolute) to the configuration file. Note that this robot must only have spheres for collision geometry.
- The end-effector to use for attachments
- Collision checking resolution
- Output template and sub-templates to use
- Output filename

The optional `dynamics_urdf` field selects a separate URDF for forward dynamics tracing.
Use it when the collision URDF has simplified or removed inertial data.
The dynamics URDF must describe the same configuration and velocity dimensions as `urdf`.

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

For a custom robot, you only need to change the `name`, `urdf`, `srdf`, and `end_effector` fields.
Some notes:
- If your robot does not have an SRDF, then the script will attempt to guess the self-collisions of the robot by randomly sampling one million configurations and seeing what is always in collision and what never collides. However, this is unreliable and probably should just be done by hand.
- If you do not provide an end-effector, the last frame in the robot will be used. For serial link manipulators, this is probably the tool frame, but you should set this yourself.
- `end_effector` may also be a list of frame names for robots with several end-effectors (e.g. a bimanual robot). The first one is the primary end-effector (the `end_effector` name constant). The generated `eefk(x, i)` returns the pose of end-effector `i` (default 0), and `fkcc_attach` poses attachments on each end-effector with `set_attachment_pose(environment, i, pose)`, checks each end-effector's attachments against the frames it can collide with, and checks attachments on different end-effectors against each other.
- Attachments are checked against the links their end-effector is self-collision checked against. The generated header also exposes `link_names` (frame names; the index is the link id) and `exempt_link_pairs` (every pair of links that is not self-collision checked, each link with itself included). With an SRDF these are exactly its `disable_collisions` pairs, so vamp can let an attachment touch a given link plus everything that link may touch (`Attachment.allowed_links`, robot module `exempt_links(names)`). Without an SRDF the collision pairs are guessed by sampling and the table also contains pairs that merely never collided, so only use it with an SRDF.
- `compact_collisions` (default `false`, C++ template only): instead of unrolling every collision check into straight-line code, emit constant tables of sphere indices and loop over them. The generated header is much smaller and compiles faster, which matters for robots with many spheres; the collision results are the same.
- `skip_static_environment_collisions` (default `false`): do not check links that no mobile joint connects to the world (fixed to the base) against the environment. Self-collision checks are unaffected.
- You may need to increase the collision checking resolution used (`resolution`) if your robot is especially high-dimensional (e.g., for Panda and Fetch 32 is used, for Baxter, 64).

## Using with VAMP

If you have a custom robot, you need to first spherize the robot (that is, convert the URDF collision geometry to only be spheres) either manually or using a tool like [foam](https://github.com/CoMMALab/foam/).
Then, write your configuration file as described above.
To then generate the required code for VAMP, simply use the provided `fkcc_gen` script with your desired robot configuration, e.g., for the Franka Panda:
```bash
./build/fkcc_gen panda.json

# Optionally format the code
clang-format -i panda_fk.hh
```

Then, copy this code into the `src/impl/vamp/robots/` folder in VAMP, where the other `<robot>.hh` files are:
```bash
# Assuming you have cloned the VAMP repo and it is located at ${VAMP_DIR}
cp panda_fk.hh ${VAMP_DIR}/src/impl/vamp/robots/panda.fk
```
Add your robot's name (the `name` field of the configuration JSON - note that this cannot have invalid characters for C++ struct names like `-` as it's used as the name of the struct) and the name of the `.hh` file (the `<robot>` part in `<robot>.hh`) to VAMP's `pyproject.toml`.
For the Panda, this would be:
```
VAMP_ROBOT_MODULES="panda"
VAMP_ROBOT_STRUCTS="Panda"
```
The name of the header file goes in `VAMP_ROBOT_MODULES`, and the name of the robot goes into `VAMP_ROBOT_STRUCTS`.
These are both lists that are separated by semicolons (`;`), you can add or remove whatever robots you want.
The name of your robot used in VAMP is the name of the header file (e.g., `panda`, or `ur5`); this is the "name" of the robot.

Once the header file is added and the name of the robot is added to the TOML, recompile VAMP:
```bash
pip install -e ${VAMP_DIR}
```

If you want to visualize your robot with the example scripts in VAMP, you will need to copy your robot's resources to VAMP's `resources/` folder.
For example, the Panda's URDF is stored in the `panda/` folder as `panda_spherized.urdf`.
If you've done this, now test your robot out with `scripts/random_dance.py --robot <robot>`.
For the Panda, this would be:
```bash
python scripts/random_dance.py --robot panda
```

## Available Parameters for Templates

Templating is done with [inja](https://github.com/pantor/inja).
Some templates are available in the `resources/templates` folder.
In addition to the specified input fields from the configuration file, the script provides the following output information usable in your templates:
- `n_q`: number of joint DoF.
- `n_spheres`: number of collision spheres.
- `bound_lower`: lower bound of joint ranges.
- `bound_range`: range between lower and upper bound of joint ranges.
- `measure`: total measure of robot joint space.
- `end_effector_index`: frame index of the first (primary) end-effector.
- `end_effectors`: names of all end-effectors, in declaration order; `num_end_effectors`: how many.
- `min_radius`: minimum sphere radius on robot.
- `max_radius`: maximum sphere radius on robot.
- `min_radius_mobile`: minimum sphere radius among mobile (non-fixed) spheres, i.e. spheres whose kinematic chain includes at least one moving joint.
- `max_radius_mobile`: maximum sphere radius among mobile (non-fixed) spheres.
- `min_bounding_radius_mobile`: minimum radius among the per-link bounding spheres (used in the collision-checking hierarchy) belonging to mobile (non-fixed) links.
- `max_bounding_radius_mobile`: maximum radius among the per-link bounding spheres belonging to mobile (non-fixed) links.
- `base_position`: position of the robot's base joint (average position, in the URDF's forward-kinematics frame, of the joint(s) whose parent is directly "universe", i.e. where the fixed mounting chain hands off to the first moving degree of freedom). Not necessarily the origin.
- `joint_names`: name of joint corresponding to each DoF.
- `link_names`: name of frame corresponding to index.
- `per_link_spheres`: for each frame, the indices of the spheres associated with that frame.
- `links_with_geometry`: indices of the frames that have collision geometry.
- `bounding_sphere_index`: mapping between frame index to bounding sphere index (where bounding spheres are at the end of the sphere buffer, after `n_spheres`).
- `end_effector_collisions`: for each end-effector, which frames it can collide with.
- `environment_links`: indices (into `links_with_geometry`, in the order the template iterates them) of the links checked against the environment; this is every link with geometry unless `skip_static_environment_collisions` is set.
- `compact_env_entries`, `compact_env_body_idx`, `compact_self_entries`, `compact_self_pair_a`, `compact_self_pair_b`: the index tables behind `compact_collisions`.
- `sphere_env_skip`: per sphere, whether its environment check is skipped (used by `fkcc_debug`).
- `allowed_link_pairs`: sphere indices that are allowed to collide with each other.
- `eefk_code`, `eefk_code_vars`, `eefk_code_output`: C-style code for computing end-effector poses in position and quaternion (12 output values per end-effector, in declaration order), the number of intermediate variables used, and the number of output variables.
- `spherefk_code`, `spherefk_code_vars`, `spherefk_code_output`: C-style code for computing position of all spheres, the number of intermediate variables used, and the number of output variables.
- `ccfk_code`, `ccfk_code_vars`, `ccfk_code_output`: C-style code for computing position of all spheres and bounding spheres for collision checking, the number of intermediate variables used, and the number of output variables.
- `ccfkee_code`, `ccfkee_code_vars`, `ccfkee_code_output`: C-style code for computing position of all spheres and bounding spheres for collision checking as well as the position and quaternion for the end-effector, the number of intermediate variables used, and the number of output variables.
