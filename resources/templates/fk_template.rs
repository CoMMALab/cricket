#![expect(
    clippy::too_many_lines,
    clippy::cognitive_complexity,
    clippy::unreadable_literal,
    clippy::collapsible_if,
    clippy::excessive_precision,
    clippy::suspicious_operation_groupings
)]

use core::simd::Simd;

use carom_core::{
    cos, env::World3d, sin, sphere_environment_in_collision, sphere_sphere_self_collision,
};

pub const DIM: usize = {{n_q}};

pub const JOINT_NAMES: [&str; DIM] = ["{{join(joint_names, "\", \"")}}"];
pub const END_EFFECTOR_NAME: &str = "{{end_effector}}";

pub const BOUND_LOWER: [f32; DIM] = [{{join(bound_lower, ", ")}}];

pub const BOUND_SCALE: [f32; DIM] = [{{join(bound_range, ", ")}}];

pub const RESOLUTION: usize = {{resolution}};

pub const MIN_RADIUS: f32 = {{min_radius}};
pub const MAX_RADIUS: f32 = {{max_radius}};

#[expect(
    clippy::too_many_lines,
    clippy::cognitive_complexity,
    clippy::unreadable_literal,
    clippy::collapsible_if,
    clippy::excessive_precision,
    clippy::suspicious_operation_groupings
)]
pub fn fkcc<const L: usize>(x: &super::ConfigurationBlock<L>, environment: &World3d<f32>) -> bool {
    let mut v = [Simd::splat(0.0); {{ccfk_code_vars}}];
    let mut y = [Simd::splat(0.0); {{ccfk_code_output}}];

    {{ccfk_code}}
    {% include "ccfk" %}
    true
}
