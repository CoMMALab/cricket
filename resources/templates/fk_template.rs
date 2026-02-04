use core::simd::Simd;

use elain::{Align, Alignment};

use crate::{
    env::World3d,
    robot::{sphere_environment_in_collision, sphere_sphere_self_collision},
    cos, sin,
};

#[expect(
    non_snake_case,
    clippy::too_many_lines,
    clippy::cognitive_complexity,
    clippy::unreadable_literal,
    clippy::approx_constant,
    clippy::collapsible_if
)]
pub fn fkcc<const L: usize>(x: &super::ConfigurationBlock<L>, environment: &World3d<f32, L>) -> bool
where
    Align<L>: Alignment,
{
    let mut v = [Simd::splat(0.0); {{ccfk_code_vars}}];
    let mut y = [Simd::splat(0.0); {{ccfk_code_output}}];

    {{ccfk_code}}
    {% include "ccfk" %}
    true
}
