#![expect(
    clippy::too_many_lines,
    clippy::cognitive_complexity,
    clippy::unreadable_literal,
    clippy::collapsible_if,
    clippy::excessive_precision,
)]
#![allow(clippy::assign_op_pattern)]
#![feature(portable_simd)]

use core::simd::Simd;

use carom_core::{
    Attach, AttachValidate, Ball, Block, BlockValidate, Collide3, PosedAttachment, Robot, SimdArithmetic, cos, sin,
    sphere_environment_in_collision, sphere_sphere_self_collision, Isometry,
};

#[derive(Clone, Copy, Debug)]
pub struct {{name}};

const DIM: usize = {{n_q}};

impl {{name}} {
    /// The configuration-space dimension of this robot.
    pub const DIM: usize = DIM;

    pub const BOUNDS: [[f32; DIM]; 2] = [
        Self::BOUND_LOWER,
        make_upper(Self::BOUND_LOWER, Self::BOUND_SCALE),
    ];

    pub const STEP_SIZE: f32 = 1.0 / {{resolution}}f32;

    pub const JOINT_NAMES: [&str; DIM] = ["{{join(joint_names, "\", \"")}}"];
    pub const END_EFFECTOR_NAME: &str = "{{end_effector}}";

    const BOUND_LOWER: [f32; DIM] = [{{join(bound_lower, ", ")}}];
    const BOUND_SCALE: [f32; DIM] = [{{join(bound_range, ", ")}}];

    pub const MIN_RADIUS: f32 = {{min_radius}};
    pub const MAX_RADIUS: f32 = {{max_radius}};
}

const fn make_upper(lower: [f32; DIM], scale: [f32; DIM]) -> [f32; DIM] {
    let mut ret = [0.0; DIM];
    let mut i = 0;
    while i < DIM {
        ret[i] = lower[i] + scale[i];
        i += 1;
    }
    ret
}

type ConfigurationBlock<const L: usize> = [Simd<f32, L>; {{name}}::DIM];

impl Robot<DIM, f32> for {{name}}
{
    fn bounds(&self) -> [[f32; DIM]; 2] {
        Self::BOUNDS
    }

    fn name(&self) -> &'static str {
        "{{lower(name)}}"
    }

    fn joint_names(&self) -> &[&str; DIM] {
        &Self::JOINT_NAMES
    }

    fn sphere_fk<const L: usize>(
        &self,
        cfgs: &Block<{ Self::DIM }, L, f32>,
        spheres: &mut Vec<Ball<3, Simd<f32, L>>>,
    ) {
        sphere_fk(&cfgs.0, spheres);
    }
}

impl<W> BlockValidate<DIM, f32, W> for {{name}}
where
    W: Collide3<f32>,
{
    fn is_valid<const L: usize>(&self, cfgs: &Block<{ Self::DIM }, L, f32>, world: &W) -> bool
    where
        Simd<f32, L>: SimdArithmetic<f32, L>,
    {
        fkcc(&cfgs.0, world)
    }
}

impl Attach<DIM, f32> for {{name}}
{
    fn eefk<const L: usize>(&self, cfgs: &Block<DIM, L, f32>) -> Isometry<Simd<f32, L>, 3> {
        eefk(&cfgs.0)
    }
}

impl<W> AttachValidate<DIM, f32, W> for {{name}}
where
    W: Collide3<f32>,
{
    fn is_valid_attach<const L: usize>(
        &self,
        cfgs: &Block<DIM, L, f32>,
        world: &W,
        attachment: &mut PosedAttachment<f32, L>,
    ) -> bool
    where
        Simd<f32, L>: crate::SimdArithmetic<f32, L>,
    {
        attach_fkcc(&cfgs.0, world, attachment)
    }
}


fn fkcc<const L: usize>(x: &ConfigurationBlock<L>, environment: &impl Collide3<f32>) -> bool {
    let mut v = [Simd::splat(0.0); {{ccfk_code_vars}}];
    let mut y = [Simd::splat(0.0); {{ccfk_code_output}}];

    {{ccfk_code}}
    {% include "ccfk" %}
    true
}


fn attach_fkcc<const L: usize>(
    x: &ConfigurationBlock<L>,
    environment: &impl Collide3<f32>,
    attachment: &mut PosedAttachment<f32, L>,
) -> bool {
    let mut v = [Simd::splat(0.0); {{ccfkee_code_vars}}];
    let mut y = [Simd::splat(0.0); {{ccfkee_code_output}}];

    {{ccfkee_code}}
    {% include "ccfk" %}

    // attaching at {{ end_effector }}
    attachment.set_pose(&Isometry::from_carom_buf(
        *y[{{ccfkee_code_output - 12}}..{{ccfkee_code_output}}].as_array().unwrap()
    ));

    //
    // attachment vs. environment collisions
    //
    if attachment.environment_collision(environment)
    {
        return false;
    }

    //
    // attachment vs. robot collisions
    //

    {% for i in range(length(end_effector_collisions)) %}
    {% set link_index = at(end_effector_collisions, i) %}
    {% set link_bs = at(bounding_sphere_index, link_index) %}
    {% set link_spheres = at(per_link_spheres, link_index) %}

    // Attachment vs. {{ at(link_names, link_index )}}
    if attachment.sphere_collision(
        y[{{(n_spheres + link_bs) * 4 + 0}}],
        y[{{(n_spheres + link_bs) * 4 + 1}}],
        y[{{(n_spheres + link_bs) * 4 + 2}}],
        y[{{(n_spheres + link_bs) * 4 + 3}}]
    ) {
        {% for j in range(length(link_spheres)) %}
        {% set sphere_index = at(link_spheres, j) %}
        if attachment.sphere_collision(
            y[{{sphere_index * 4 + 0}}],
            y[{{sphere_index * 4 + 1}}],
            y[{{sphere_index * 4 + 2}}],
            y[{{sphere_index * 4 + 3}}]
        ) {
            return false;
        }
        {% endfor %}
    }
    {% endfor %}

    true
}

fn sphere_fk<const L: usize>(x: &ConfigurationBlock<L>, spheres: &mut Vec<Ball<3, Simd<f32, L>>>)
{
    let mut v = [Simd::splat(0.0); {{spherefk_code_vars}}];
    let mut y = [Simd::splat(0.0); {{spherefk_code_output}}];

    {{spherefk_code}}

    for &[x, y, z, r] in y.as_chunks().0 {
        spheres.push(Ball {
            pos: [x, y, z],
            r,
        });
    }
}


fn eefk<const L: usize>(x: &ConfigurationBlock<L>) -> Isometry<Simd<f32, L>, 3>
{
    let mut v = [Simd::splat(0.0); {{eefk_code_vars}}];
    let mut y = [Simd::splat(0.0); {{eefk_code_output}}];

    {{eefk_code}}

    Isometry::from_carom_buf(y)
}
