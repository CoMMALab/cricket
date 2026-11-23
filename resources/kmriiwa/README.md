Files downloaded from [stoic-roboticist/kmriiwa_ros_stack](https://github.com/stoic-roboticist/kmriiwa_ros_stack).

Various modifications were used to convert from the raw xacros to a model usable in drake, including:

- Disinfecting the files of as much ROS and Gazebo influence as possible.
- Cleaned up and simplified file paths.
- Compiling the `xacro` files to `urdf`.
- Converting meshes to `gltf` and `obj`.
    - See `meshes/dae_to_gltf.sh`, `meshes/dae_to_obj.sh`, and `meshes/recursive_to_obj.sh`.
    - The orientation of the `obj` files is incorrect, so we use `meshes/fix_orientation.py` to correct this.
    - The material properties for meshes/kmp200.
- Removed unnecessary collision geometries (and simplified the base collision geometry).