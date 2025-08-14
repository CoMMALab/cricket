# Cricket: Tracing Compilation for Spherized Robots

The parts I changed were -
- panda_main_templace.hh: env_collision_check (1) checks sphere in reverse order, and (2) enables per warp early exit through warp primitives
- panda_main_templace.hh: self_collision_check (1) enables per warp early exit through warp primitives

The above changes are only made to the detailed env and self collision check. I did not modify the approx template. I also did not touch the fk function so hopefully should not conflict with the recent changes for Baxter.
