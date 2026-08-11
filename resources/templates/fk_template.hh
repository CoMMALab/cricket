#pragma once

#include <vamp/vector.hh>
#include <vamp/vector/math.hh>
#include <vamp/collision/environment.hh>
#include <vamp/collision/validity.hh>
{% if has_flask %}#include <vamp/planning/flask.hh>{% endif %}

#include <Eigen/Geometry>
{% if has_flask %}
#include <algorithm>
#include <cmath>
#include <limits>
{% endif %}

// clang-format off
// NOLINTBEGIN(*-magic-numbers)
namespace vamp::robots
{
struct {{name}}
{
    static constexpr const char *name = "{{module_name}}";
    static constexpr std::size_t dimension = {{n_q}};
    static constexpr std::size_t sample_dimension = {{n_u}};
    static constexpr std::size_t n_spheres = {{n_spheres}};
    static constexpr float min_radius = {{min_radius}};
    static constexpr float max_radius = {{max_radius}};
    static constexpr std::size_t resolution = {{resolution}};
    static constexpr bool euclidean = {{euclidean}};
    static constexpr std::array<std::size_t, {{length(so3_offsets)}}> so3_offsets = { {{join(so3_offsets, ", ")}} };

    static constexpr std::array<std::string_view, dimension> joint_names = {"{{join(joint_names, "\", \"")}}"};
    static constexpr const char *end_effector = "{{end_effector}}";

    using Configuration = FloatVector<dimension>;
    struct alignas(FloatVectorAlignment) ConfigurationArray
        : std::array<FloatT, dimension>
    {
    };
    using Sample = FloatVector<sample_dimension>;

    struct alignas(FloatVectorAlignment) ConfigurationBuffer
        : std::array<float, Configuration::num_scalars_rounded>
    {
    };

    template <std::size_t rake>
    using ConfigurationBlock = FloatVector<rake, dimension>;

    using State = Configuration;
    using StateArray = ConfigurationArray;
    using StateBuffer = ConfigurationBuffer;
    template <std::size_t rake>
    using StateBlock = ConfigurationBlock<rake>;

    template <std::size_t rake>
    struct Spheres
    {
        FloatVector<rake, n_spheres> x;
        FloatVector<rake, n_spheres> y;
        FloatVector<rake, n_spheres> z;
        FloatVector<rake, n_spheres> r;
    };

    alignas(Configuration::S::Alignment) static constexpr std::array<float, dimension> s_m{
        {{join(bound_range, ", ")}}
    };

    alignas(Configuration::S::Alignment) static constexpr std::array<float, dimension> s_a{
        {{join(bound_lower, ", ")}}
    };

    alignas(Configuration::S::Alignment) static constexpr std::array<float, dimension> d_m{
        {{join(bound_descale, ", ")}}
    };

    static inline void scale_configuration(Configuration& q) noexcept
    {
        q = q * Configuration(s_m) + Configuration(s_a);
    }

    static inline void descale_configuration(Configuration& q) noexcept
    {
        q = (q - Configuration(s_a)) * Configuration(d_m);
    }

    template <std::size_t rake>
    static inline void scale_configuration_block(ConfigurationBlock<rake> &q) noexcept
    {
        {% for index in range(n_q) -%}
        q[{{index}}] = {{ at(bound_lower, index) }} + (q[{{index}}] * {{ at(bound_range, index) }});
        {%- endfor %}
    }

    template <std::size_t rake>
    static inline void descale_configuration_block(ConfigurationBlock<rake> & q) noexcept
    {
        {% for index in range(n_q) -%}
        q[{{index}}] = {{ at(bound_descale, index) }} * (q[{{index}}] - {{ at(bound_lower, index) }});
        {%- endfor %}
    }

    inline static auto space_measure() noexcept -> float
    {
        return {{measure}};
    }

    alignas(Configuration::S::Alignment) static constexpr std::array<float, dimension> lower_bound{
        {{join(lower, ", ")}}
    };

    alignas(Configuration::S::Alignment) static constexpr std::array<float, dimension> upper_bound{
        {{join(upper, ", ")}}
    };

    static inline auto in_bounds(const Configuration &x) -> bool
    {
        return (x <= Configuration(upper_bound)).all() and (x >= Configuration(lower_bound)).all();
    }

    static inline auto sample(const Sample &x_in) -> Configuration
    {
        {% if euclidean %}
        // Euclidean fast path: same affine map as scale_configuration,
        // operating on the packed FloatVector in one SIMD step. Sample and
        // Configuration alias to the same FloatVector type when Euclidean.
        Configuration q = x_in;
        scale_configuration(q);
        return q;
        {% else %}
        {% if mapconfig_code_vars > 0 %}std::array<float, {{mapconfig_code_vars}}> v;{% endif %}
        // Value-init: pad lanes past `dimension` must be zero, since Configuration
        // loads the full rounded width and full-width reductions (e.g. squared_l2_norm)
        // include them.
        ConfigurationBuffer y{};
        const auto x = x_in.to_array();
        {{mapconfig_code}}
        return Configuration(y.data());
        {% endif %}
    }

    static inline auto distance(const Configuration &a_in, const Configuration &b_in) -> float
    {
        {% if euclidean %}
        return a_in.distance(b_in);
        {% else %}
        {% if distance_code_vars > 0 %}std::array<float, {{distance_code_vars}}> v;{% endif %}
        std::array<float, 1> y;
        const auto a = a_in.to_array();
        const auto b = b_in.to_array();
        {{distance_code}}
        return y[0];
        {% endif %}
    }

    static inline auto interpolate(const Configuration &a_in, const Configuration &b_in, float t) -> Configuration
    {
        {% if euclidean %}
        return a_in.interpolate(b_in, t);
        {% else %}
        {% if interpolate_code_vars > 0 %}std::array<float, {{interpolate_code_vars}}> v;{% endif %}
        // Value-init: pad lanes past `dimension` must be zero, since Configuration
        // loads the full rounded width and full-width reductions (e.g. squared_l2_norm)
        // include them.
        ConfigurationBuffer y{};
        const auto a = a_in.to_array();
        const auto b = b_in.to_array();
        {{interpolate_code}}
        return Configuration(y.data());
        {% endif %}
    }

    template <std::size_t rake>
    static inline void interpolate_block(
        const Configuration &a,
        const Configuration &b,
        const FloatVector<rake> &t,
        ConfigurationBlock<rake> &out) noexcept
    {
        // V is referenced by the SIMD-mask form emitted by LanguageCVampBlock
        // for non-Euclidean robots; constants and operands get wrapped as
        // V(x).blend(V(y), (V(...) CMP V(...))) so .blend() always has a
        // vector receiver.
        using V = FloatVector<rake, 1>;
        {% if interpolate_block_code_vars > 0 %}std::array<V, {{interpolate_block_code_vars}}> v;{% endif %}
        {{interpolate_block_code}}
    }

    // The FK/CC kernels read only rows 0..dimension-1, so they accept any block with at least
    // that many rows; a nested FLASK robot forwards its 3n-row z-blocks here directly.
    template <std::size_t rake, std::size_t stride = dimension>
    static inline void sphere_fk(const FloatVector<rake, stride> &x, Spheres<rake> &out) noexcept
    {
        static_assert(stride >= dimension);
        std::array<FloatVector<rake, 1>, {{spherefk_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{spherefk_code_output}}> y;

        {{spherefk_code}}

        for (auto i = 0U; i < {{n_spheres}}; ++i)
        {
            out.x[i] = y[i * 4 + 0];
            out.y[i] = y[i * 4 + 1];
            out.z[i] = y[i * 4 + 2];
            out.r[i] = y[i * 4 + 3];
        }
    }

    using Debug = std::pair<std::vector<std::vector<std::string>>, std::vector<std::pair<std::size_t, std::size_t>>>;

    template <std::size_t rake, std::size_t stride = dimension>
        static inline auto fkcc_debug(
            const vamp::collision::Environment<FloatVector<rake>> &environment,
            const FloatVector<rake, stride> &x) noexcept -> Debug
    {
        static_assert(stride >= dimension);
        std::array<FloatVector<rake, 1>, {{ccfk_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{ccfk_code_output}}> y;

        {{ccfk_code}}

        Debug output;

        {% for i in range(n_spheres) %}{% if at(sphere_env_skip, i) %}
        output.first.emplace_back();{% else %}
        output.first.emplace_back(
            sphere_environment_get_collisions<decltype(x[0])>(
                environment,
                y[{{ i * 4 + 0 }}],
                y[{{ i * 4 + 1 }}],
                y[{{ i * 4 + 2 }}],
                y[{{ i * 4 + 3 }}]));{% endif %}
        {% endfor %}

        {% if compact_collisions %}
        for (std::size_t i = 0; i < cc_self_pair_a.size(); ++i)
        {
            const auto a = cc_self_pair_a[i] * 4;
            const auto b = cc_self_pair_b[i] * 4;
            if (sphere_sphere_self_collision<decltype(x[0])>(
                    y[a + 0], y[a + 1], y[a + 2], y[a + 3],
                    y[b + 0], y[b + 1], y[b + 2], y[b + 3]))
            {
                output.second.emplace_back(cc_self_pair_a[i], cc_self_pair_b[i]);
            }
        }
        {% else %}
        {% for i in range(length(allowed_link_pairs)) %}
        {% set pair = at(allowed_link_pairs, i) %}
        {% set link_1_index = at(pair, 0) %}
        {% set link_2_index = at(pair, 1) %}
        {% set link_1_spheres = at(per_link_spheres, link_1_index) %}
        {% set link_2_spheres = at(per_link_spheres, link_2_index) %}

        {% for j in range(length(link_1_spheres)) %}
        {% for k in range(length(link_2_spheres)) %}

        {% set sphere_1_loc = at(link_1_spheres, j) %}
        {% set sphere_2_loc = at(link_2_spheres, k) %}

        if (sphere_sphere_self_collision<decltype(x[0])>(y[{{ sphere_1_loc * 4 + 0}} ],
                                                         y[{{ sphere_1_loc * 4 + 1}} ],
                                                         y[{{ sphere_1_loc * 4 + 2}} ],
                                                         y[{{ sphere_1_loc * 4 + 3}} ],
                                                         y[{{ sphere_2_loc * 4 + 0}} ],
                                                         y[{{ sphere_2_loc * 4 + 1}} ],
                                                         y[{{ sphere_2_loc * 4 + 2}} ],
                                                         y[{{ sphere_2_loc * 4 + 3}} ]))
        {
            output.second.emplace_back({{ sphere_1_loc }}, {{ sphere_2_loc }});
        }

        {% endfor %}
        {% endfor %}
        {% endfor %}
        {% endif %}

        return output;
    }

    template <std::size_t rake, std::size_t stride = dimension>
        static inline bool fkcc(
            const vamp::collision::Environment<FloatVector<rake>> &environment,
            const FloatVector<rake, stride> &x) noexcept
    {
        static_assert(stride >= dimension);
        std::array<FloatVector<rake, 1>, {{ccfk_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{ccfk_code_output}}> y;

        {{ccfk_code}}
        {% include "ccfk" %}

        return true;
    }

    template <std::size_t rake, std::size_t stride = dimension>
    static inline bool fkcc_attach(
        const vamp::collision::Environment<FloatVector<rake>> &environment,
        const FloatVector<rake, stride> &x) noexcept
    {
        static_assert(stride >= dimension);
        std::array<FloatVector<rake, 1>, {{ccfkee_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{ccfkee_code_output}}> y;

        {{ccfkee_code}}
        {% include "ccfk" %}

        {% for k in range(num_end_effectors) %}
        // attaching at {{ at(end_effectors, k) }}
        set_attachment_pose(environment, {{k}}, to_isometry(&y[{{ccfkee_code_output - 12 * (num_end_effectors - k)}}]));
        {% endfor %}

        //
        // attachment vs. environment collisions
        //
        if (attachment_environment_collision(environment)) [[unlikely]]
        {
            return false;
        }
        {% if num_end_effectors > 1 %}

        //
        // attachment vs. attachment collisions (across end-effectors)
        //
        if (attachment_attachment_collision(environment)) [[unlikely]]
        {
            return false;
        }
        {% endif %}

        //
        // attachment vs. robot collisions
        //

        {% for k in range(num_end_effectors) %}
        {% set eef_collisions = at(end_effector_collisions, k) %}
        {% for i in range(length(eef_collisions)) %}
        {% set link_index = at(eef_collisions, i) %}
        {% set link_bs = at(bounding_sphere_index, link_index) %}
        {% set link_spheres = at(per_link_spheres, link_index) %}

        // {{ at(end_effectors, k) }} attachments vs. {{ at(link_names, link_index )}}
        if (attachment_sphere_collision<decltype(x[0])>(environment, {{k}},
                                                        y[{{(n_spheres + link_bs) * 4 + 0}}],
                                                        y[{{(n_spheres + link_bs) * 4 + 1}}],
                                                        y[{{(n_spheres + link_bs) * 4 + 2}}],
                                                        y[{{(n_spheres + link_bs) * 4 + 3}}])) [[unlikely]]
        {
            {% for j in range(length(link_spheres)) %}
            {% set sphere_index = at(link_spheres, j) %}
            if (attachment_sphere_collision<decltype(x[0])>(environment, {{k}},
                                                            y[{{sphere_index * 4 + 0}}],
                                                            y[{{sphere_index * 4 + 1}}],
                                                            y[{{sphere_index * 4 + 2}}],
                                                            y[{{sphere_index * 4 + 3}}])) [[unlikely]]
            {
                return false;
            }
            {% endfor %}
        }
        {% endfor %}
        {% endfor %}

        return true;
    }

    static inline auto eefk(const std::array<float, {{n_q}}> &x, std::size_t eef_index = 0) noexcept -> Eigen::Isometry3f
    {
        std::array<float, {{eefk_code_vars}}> v;
        std::array<float, {{eefk_code_output}}> y;

        {{eefk_code}}

        return to_isometry(y.data() + 12 * eef_index);
    }

    {% if has_constraints %}
    //
    // TSR (task-space region) constraint functions
    //

    static constexpr std::size_t n_eef = {{num_end_effectors}};
    static constexpr std::array<std::string_view, {{num_end_effectors}}> end_effectors = {"{{join(end_effectors, "\", \"")}}"};

    // Input: q (dimension), then per end-effector rTe (7), wTr (7), lb (6), ub (6); transforms
    // are wxyz quaternion + xyz translation. Output: d(err)/dq (6 * n_eef * dimension,
    // row-major), then the raw error (6 * n_eef).
    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto tsr_error(const InputVector &x, OutputVector &out) noexcept
    {
        std::array<FloatVector<rake, 1>, {{tsr_error_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{tsr_error_code_output}}> y;

        {{tsr_error_code}}

        for (auto i = 0U; i < {{tsr_error_code_output}}; ++i)
        {
            out[i] = y[i];
        }
    }

    // Input for the solvers: J (6 * n_eef * dimension, row-major), then err (6 * n_eef).
    template <std::size_t rake, typename InputVector>
    static inline auto solve_tsr_error_lm_inner(const InputVector &x, ConfigurationBlock<rake> &y) noexcept
    {
        std::array<FloatVector<rake, 1>, {{solve_tsr_error_lm_inner_code_vars}}> v;

        {{solve_tsr_error_lm_inner_code}}
    }

    template <std::size_t rake, typename InputVector>
    static inline auto solve_tsr_error_lm_outer(const InputVector &x, ConfigurationBlock<rake> &y) noexcept
    {
        std::array<FloatVector<rake, 1>, {{solve_tsr_error_lm_outer_code_vars}}> v;

        {{solve_tsr_error_lm_outer_code}}
    }

    template <std::size_t rake, typename InputVector>
    static inline auto solve_tsr_error_gradient_descent(const InputVector &x, ConfigurationBlock<rake> &y) noexcept
    {
        std::array<FloatVector<rake, 1>, {{solve_tsr_error_gradient_descent_code_vars}}> v;

        {{solve_tsr_error_gradient_descent_code}}
    }

    {% if num_end_effectors > 1 %}
    // Relative-pose TSR between the first two end-effectors.
    // Input: q (dimension), then the reference relative transform lTr (7), lb (6), ub (6).
    // Output: d(err)/dq (6 * dimension, row-major), then the raw error (6).
    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto tsr_bimanual_error(const InputVector &x, OutputVector &out) noexcept
    {
        std::array<FloatVector<rake, 1>, {{tsr_bimanual_error_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{tsr_bimanual_error_code_output}}> y;

        {{tsr_bimanual_error_code}}

        for (auto i = 0U; i < {{tsr_bimanual_error_code_output}}; ++i)
        {
            out[i] = y[i];
        }
    }

    // Input for the relative solvers: J (6 * dimension, row-major), then err (6).
    template <std::size_t rake, typename InputVector>
    static inline auto solve_tsr_relative_error_lm_inner(const InputVector &x, ConfigurationBlock<rake> &y) noexcept
    {
        std::array<FloatVector<rake, 1>, {{solve_tsr_relative_error_lm_inner_code_vars}}> v;

        {{solve_tsr_relative_error_lm_inner_code}}
    }

    template <std::size_t rake, typename InputVector>
    static inline auto solve_tsr_relative_error_lm_outer(const InputVector &x, ConfigurationBlock<rake> &y) noexcept
    {
        std::array<FloatVector<rake, 1>, {{solve_tsr_relative_error_lm_outer_code_vars}}> v;

        {{solve_tsr_relative_error_lm_outer_code}}
    }

    template <std::size_t rake, typename InputVector>
    static inline auto solve_tsr_relative_error_gradient_descent(const InputVector &x, ConfigurationBlock<rake> &y) noexcept
    {
        std::array<FloatVector<rake, 1>, {{solve_tsr_relative_error_gradient_descent_code_vars}}> v;

        {{solve_tsr_relative_error_gradient_descent_code}}
    }
    {% endif %}
    {% endif %}

    {% if has_com %}
    //
    // Center-of-mass kinematics
    //

    static constexpr bool has_com = true;

    // Input: q (dimension). Output: d(com)/dq (3 * dimension, row-major), then the center of
    // mass (3){% if length(com_reference_frames) > 0 %}, relative to the mean position of: {{join(com_reference_frames, ", ")}}{% endif %}.
    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto com_jacobian(const InputVector &x, OutputVector &out) noexcept
    {
        std::array<FloatVector<rake, 1>, {{com_jacobian_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{com_jacobian_code_output}}> y;

        {{com_jacobian_code}}

        for (auto i = 0U; i < {{com_jacobian_code_output}}; ++i)
        {
            out[i] = y[i];
        }
    }

    // Input for the solvers: J (2 * dimension, row-major), then err (2), the xy
    // support-polygon error of the center of mass.
    template <std::size_t rake, typename InputVector>
    static inline auto solve_com_error_lm_inner(const InputVector &x, ConfigurationBlock<rake> &y) noexcept
    {
        std::array<FloatVector<rake, 1>, {{solve_com_error_lm_inner_code_vars}}> v;

        {{solve_com_error_lm_inner_code}}
    }

    template <std::size_t rake, typename InputVector>
    static inline auto solve_com_error_lm_outer(const InputVector &x, ConfigurationBlock<rake> &y) noexcept
    {
        std::array<FloatVector<rake, 1>, {{solve_com_error_lm_outer_code_vars}}> v;

        {{solve_com_error_lm_outer_code}}
    }

    template <std::size_t rake, typename InputVector>
    static inline auto solve_com_error_gradient_descent(const InputVector &x, ConfigurationBlock<rake> &y) noexcept
    {
        std::array<FloatVector<rake, 1>, {{solve_com_error_gradient_descent_code_vars}}> v;

        {{solve_com_error_gradient_descent_code}}
    }
    {% endif %}

    {% if has_closed_loops %}
    //
    // Loop-closure distance constraints
    //

    static constexpr std::size_t n_closed_loops = {{num_closed_loops}};

    // Input: q (dimension). Output: d(err)/dq (n_closed_loops * dimension, row-major), then
    // err (n_closed_loops), where each row is the deviation of the distance between the
    // loop's cut frames from its fixed length.
    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto closed_loop_error(const InputVector &x, OutputVector &out) noexcept
    {
        std::array<FloatVector<rake, 1>, {{closed_loop_error_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{closed_loop_error_code_output}}> y;

        {{closed_loop_error_code}}

        for (auto i = 0U; i < {{closed_loop_error_code_output}}; ++i)
        {
            out[i] = y[i];
        }
    }

    // Input for the solvers: J (n_closed_loops * dimension, row-major), then err
    // (n_closed_loops).
    template <std::size_t rake, typename InputVector>
    static inline auto solve_closed_loop_error_lm_inner(const InputVector &x, ConfigurationBlock<rake> &y) noexcept
    {
        std::array<FloatVector<rake, 1>, {{solve_closed_loop_error_lm_inner_code_vars}}> v;

        {{solve_closed_loop_error_lm_inner_code}}
    }

    template <std::size_t rake, typename InputVector>
    static inline auto solve_closed_loop_error_lm_outer(const InputVector &x, ConfigurationBlock<rake> &y) noexcept
    {
        std::array<FloatVector<rake, 1>, {{solve_closed_loop_error_lm_outer_code_vars}}> v;

        {{solve_closed_loop_error_lm_outer_code}}
    }

    template <std::size_t rake, typename InputVector>
    static inline auto solve_closed_loop_error_gradient_descent(const InputVector &x, ConfigurationBlock<rake> &y) noexcept
    {
        std::array<FloatVector<rake, 1>, {{solve_closed_loop_error_gradient_descent_code_vars}}> v;

        {{solve_closed_loop_error_gradient_descent_code}}
    }
    {% endif %}

    {% if has_lead_screw %}
    //
    // Lead-screw coupling constraint functions
    //

    static constexpr bool has_lead_screw = true;

    // Input: q (dimension), then rTe (7), wTr (7), pitch (1); transforms are wxyz
    // quaternion + xyz translation. Output: d(h)/dq (dimension, row-major), then h (1):
    // the axial advance of the offset end-effector frame along wTr's z-axis minus the
    // pitch-scaled rotation about it. dh/dq is the Pfaffian row a(q)^T of the velocity
    // coupling a(q)^T qdot = 0; h itself is the conserved quantity of the integrable
    // (holonomic) form.
    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto lead_screw_error(const InputVector &x, OutputVector &out) noexcept
    {
        std::array<FloatVector<rake, 1>, {{lead_screw_error_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{lead_screw_error_code_output}}> y;

        {{lead_screw_error_code}}

        for (auto i = 0U; i < {{lead_screw_error_code_output}}; ++i)
        {
            out[i] = y[i];
        }
    }

    // Input for the solvers: J (dimension, row-major), then err (1).
    template <std::size_t rake, typename InputVector>
    static inline auto solve_lead_screw_error_lm_inner(const InputVector &x, ConfigurationBlock<rake> &y) noexcept
    {
        std::array<FloatVector<rake, 1>, {{solve_lead_screw_error_lm_inner_code_vars}}> v;

        {{solve_lead_screw_error_lm_inner_code}}
    }

    template <std::size_t rake, typename InputVector>
    static inline auto solve_lead_screw_error_lm_outer(const InputVector &x, ConfigurationBlock<rake> &y) noexcept
    {
        std::array<FloatVector<rake, 1>, {{solve_lead_screw_error_lm_outer_code_vars}}> v;

        {{solve_lead_screw_error_lm_outer_code}}
    }

    template <std::size_t rake, typename InputVector>
    static inline auto solve_lead_screw_error_gradient_descent(const InputVector &x, ConfigurationBlock<rake> &y) noexcept
    {
        std::array<FloatVector<rake, 1>, {{solve_lead_screw_error_gradient_descent_code_vars}}> v;

        {{solve_lead_screw_error_gradient_descent_code}}
    }
    {% endif %}

    {% if has_twist %}
    //
    // Twist Jacobians for constant-coefficient Pfaffian velocity constraints
    //

    static constexpr bool has_twist = true;

    // Input: q (dimension), then rTe (7), wTr (7); transforms are wxyz quaternion + xyz
    // translation. Output (12 x dimension, row-major): the twist Jacobian [linear;
    // angular] of the offset frame (eef * rTe^-1) expressed in the reference frame wTr's
    // axes, then the same expressed in the offset frame's own (body) axes. Purely
    // geometric (no log map), so rows are smooth for unbounded rotation.
    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto twist_jacobians(const InputVector &x, OutputVector &out) noexcept
    {
        std::array<FloatVector<rake, 1>, {{twist_jacobians_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{twist_jacobians_code_output}}> y;

        {{twist_jacobians_code}}

        for (auto i = 0U; i < {{twist_jacobians_code_output}}; ++i)
        {
            out[i] = y[i];
        }
    }
    {% endif %}

    {% if compact_collisions %}
    struct CCEnvLink { unsigned int bs_array_idx; unsigned int body_start; unsigned int body_count; };
    struct CCSelfPair { unsigned int bs1_idx; unsigned int bs2_idx; unsigned int pair_start; unsigned int pair_count; };

    static constexpr std::array<CCEnvLink, {{ length(compact_env_entries) }}> cc_env_links = {
        {% for e in compact_env_entries %}CCEnvLink{ {{at(e,0)}}, {{at(e,1)}}, {{at(e,2)}} },
        {% endfor %}
    };
    static constexpr std::array<unsigned int, {{ length(compact_env_body_idx) }}> cc_env_body_idx = {
        {% for v in compact_env_body_idx %}{{v}}, {% endfor %}
    };
    static constexpr std::array<CCSelfPair, {{ length(compact_self_entries) }}> cc_self_pairs = {
        {% for e in compact_self_entries %}CCSelfPair{ {{at(e,0)}}, {{at(e,1)}}, {{at(e,2)}}, {{at(e,3)}} },
        {% endfor %}
    };
    static constexpr std::array<unsigned int, {{ length(compact_self_pair_a) }}> cc_self_pair_a = {
        {% for v in compact_self_pair_a %}{{v}}, {% endfor %}
    };
    static constexpr std::array<unsigned int, {{ length(compact_self_pair_b) }}> cc_self_pair_b = {
        {% for v in compact_self_pair_b %}{{v}}, {% endfor %}
    };
    {% endif %}

    {% if has_flask %}
{{ flask_struct }}
    {% endif %}

    {% if has_parameterized_space %}
    //
    // ParameterizedSpace: an alternative Space to plan over instead of this robot's own
    // Configuration -- e.g. a task-space parameterization whose samples resolve_block() to
    // an ambient Configuration for FK/collision-checking. Rendered into the parent robot
    // struct; the parent is the ambient configuration-space robot whose FK/CC kernels
    // resolve_block() hands off to. Exposes the same dimension/State/StateArray/StateBlock<
    // rake>/StateBuffer family as the outer struct's own default Space (see Part 1's State =
    // Configuration aliases above) so vamp's planners can be templated on either one.
    //

    struct ParameterizedSpace
    {
        static constexpr const char *name = "parameterized_space";
        static constexpr std::size_t dimension = {{param_dimension}};
        static constexpr std::size_t sample_dimension = {{param_sample_dimension}};
        static constexpr bool euclidean = {{param_euclidean}};
        static constexpr std::array<std::size_t, {{length(param_so3_offsets)}}> so3_offsets = { {{join(param_so3_offsets, ", ")}} };

        // Ambient configuration-space robot that resolve_block() maps into.
        using Ambient = {{name}};

        // GCP (branch) selectors for each rainbow arm's redundant self-motion manifold:
        // (elbow_sel, shoulder_sel, wrist_sel) -- see RainbowLeftArmParameterization /
        // RainbowRightArmParameterization in rainbow_arm_parameterization.hh. Fixed for the
        // whole planning problem (not part of State), so param_ik_code reads these class
        // members directly by name (`left_gcp[i]` / `right_gcp[i]`) instead of taking them as
        // part of resolve_block's input. Overwrite directly, e.g. from a binding as
        // `robot.parameterized.left_gcp = [...]`.
        inline static thread_local std::array<float, 3> left_gcp = {0.0f, 1.0f, 1.0f};
        inline static thread_local std::array<float, 3> right_gcp = {0.0f, 1.0f, 1.0f};

        // Fixed per-planning-problem SE3 offsets from the mid-frame T_mid to each hand (x, y,
        // z, qx, qy, qz, qw): T_l = T_mid * t_mid_left, T_r = T_mid * t_mid_right. Not part of
        // State -- param_ik_code reads these class members directly by name, same mechanism
        // as left_gcp/right_gcp above. Default identity (both hands coincide with T_mid);
        // overwrite directly or via compute_mid_pose() below.
        inline static thread_local std::array<float, 7> t_mid_left = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        inline static thread_local std::array<float, 7> t_mid_right = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};

        using State = FloatVector<dimension>;
        struct alignas(FloatVectorAlignment) StateArray
            : std::array<FloatT, dimension>
        {
        };
        using Sample = FloatVector<sample_dimension>;

        struct alignas(FloatVectorAlignment) StateBuffer
            : std::array<float, State::num_scalars_rounded>
        {
        };

        template <std::size_t rake>
        using StateBlock = FloatVector<rake, dimension>;

        static inline auto sample(const Sample &x_in) -> State
        {
            {% if param_sample_code_vars > 0 %}std::array<float, {{param_sample_code_vars}}> v;{% endif %}
            // Value-init: pad lanes past `dimension` must be zero, matching Configuration's
            // own sample()/interpolate() above.
            StateBuffer y{};
            const auto x = x_in.to_array();
            {{param_sample_code}}
            return State(y.data());
        }

        static inline auto distance(const State &a_in, const State &b_in) -> float
        {
            {% if param_distance_code_vars > 0 %}std::array<float, {{param_distance_code_vars}}> v;{% endif %}
            std::array<float, 1> y;
            const auto a = a_in.to_array();
            const auto b = b_in.to_array();
            {{param_distance_code}}
            return y[0];
        }

        static inline auto interpolate(const State &a_in, const State &b_in, float t) -> State
        {
            {% if param_interpolate_code_vars > 0 %}std::array<float, {{param_interpolate_code_vars}}> v;{% endif %}
            StateBuffer y{};
            const auto a = a_in.to_array();
            const auto b = b_in.to_array();
            {{param_interpolate_code}}
            return State(y.data());
        }

        template <std::size_t rake>
        static inline void interpolate_block(
            const State &a,
            const State &b,
            const FloatVector<rake> &t,
            StateBlock<rake> &out) noexcept
        {
            using V = FloatVector<rake, 1>;
            {% if param_interpolate_block_code_vars > 0 %}std::array<V, {{param_interpolate_block_code_vars}}> v;{% endif %}
            {{param_interpolate_block_code}}
        }

        // Derives t_mid_left/t_mid_right from a reference whole-body configuration `q`: FK to
        // both hands (ee_left, ee_right), take T_w_mid as the midpoint of their translations
        // with identity rotation, then t_mid_left/t_mid_right are each hand's pose relative to
        // that mid-frame. Scalar (non-rake) utility, called once per planning problem, not in
        // a hot loop. Overwrites t_mid_left/t_mid_right in place; callers that want a fixed
        // offset can still assign over them afterwards.
        static inline void compute_mid_pose(const Ambient::ConfigurationArray &q) noexcept
        {
            {% if param_mid_pose_fk_code_vars > 0 %}std::array<float, {{param_mid_pose_fk_code_vars}}> v;{% endif %}
            std::array<float, {{param_mid_pose_fk_code_output}}> y;
            const auto &x = q;

            {{param_mid_pose_fk_code}}

            Eigen::Isometry3f T_w_l0 = to_isometry(&y[0]);
            Eigen::Isometry3f T_w_r0 = to_isometry(&y[12]);

            Eigen::Isometry3f T_w_mid = Eigen::Isometry3f::Identity();
            T_w_mid.translation() = 0.5f * (T_w_l0.translation() + T_w_r0.translation());

            Eigen::Isometry3f T_w_mid_inv = T_w_mid.inverse();
            Eigen::Isometry3f T_mid_right_iso = T_w_mid_inv * T_w_r0;
            Eigen::Isometry3f T_mid_left_iso = T_w_mid_inv * T_w_l0;

            Eigen::Quaternionf qr(T_mid_right_iso.rotation());
            t_mid_right = {
                T_mid_right_iso.translation().x(), T_mid_right_iso.translation().y(), T_mid_right_iso.translation().z(),
                qr.x(), qr.y(), qr.z(), qr.w()};

            Eigen::Quaternionf ql(T_mid_left_iso.rotation());
            t_mid_left = {
                T_mid_left_iso.translation().x(), T_mid_left_iso.translation().y(), T_mid_left_iso.translation().z(),
                ql.x(), ql.y(), ql.z(), ql.w()};
        }

        // World-frame center-of-mass position for a reference whole-body configuration `q`.
        // Scalar (non-rake) utility, same call pattern as compute_mid_pose above; traced with
        // compute_jac=false so no Jacobian rows are computed or emitted.
        static inline auto compute_com(const Ambient::ConfigurationArray &q) noexcept -> std::array<float, 3>
        {
            {% if param_com_code_vars > 0 %}std::array<float, {{param_com_code_vars}}> v;{% endif %}
            std::array<float, {{param_com_code_output}}> y;
            const auto &x = q;

            {{param_com_code}}

            return y;
        }

        // Left/right end-effector WORLD poses (translation + rotation matrix, 24 floats total
        // -- vamp::to_isometry's 12-float layout, twice) derived from a StateBlock's T_mid
        // slice plus the fixed t_mid_left/t_mid_right offsets above: T_l = T_mid * t_mid_left,
        // T_r = T_mid * t_mid_right. Pure SE3 composition -- no FK, no arm IK, same algebra as
        // param_ik_code's mid-pose handling (see RainbowConstrainedBimanualIkCG in
        // rainbow_ik_cg.hh) but broken out standalone for callers that only need the hands'
        // world poses, e.g. eefs_in_collision below. Generated (RainbowEefWorldPosesFromMidCG),
        // branch-free, so this same code is valid for any `rake`: the generated arithmetic
        // only uses operators FloatVector<rake, 1> overloads, scalar (rake == 1) included.
        template <std::size_t rake>
        static inline auto eef_world_poses(const StateBlock<rake> &x_in) noexcept -> std::array<FloatVector<rake, 1>, 24>
        {
            using V = FloatVector<rake, 1>;

            std::array<V, 21> x{
                x_in[12], x_in[13], x_in[14], x_in[15], x_in[16], x_in[17], x_in[18],
                V(t_mid_left[0]), V(t_mid_left[1]), V(t_mid_left[2]),
                V(t_mid_left[3]), V(t_mid_left[4]), V(t_mid_left[5]), V(t_mid_left[6]),
                V(t_mid_right[0]), V(t_mid_right[1]), V(t_mid_right[2]),
                V(t_mid_right[3]), V(t_mid_right[4]), V(t_mid_right[5]), V(t_mid_right[6])};

            {% if param_eef_world_poses_code_vars > 0 %}std::array<V, {{param_eef_world_poses_code_vars}}> v;{% endif %}
            std::array<V, {{param_eef_world_poses_code_output}}> y;

            {{param_eef_world_poses_code}}

            return y;
        }

        {% if num_end_effectors == 2 %}
        // Fast partial collision pre-filter, batched over `rake` sampled parameterized
        // states: computes the left/right hands' world poses via eef_world_poses<rake>()
        // above (no arm IK) and checks:
        //   1. each hand's OWN rigidly-attached spheres (gripper/finger geometry baked into
        //      the URDF, see RainbowEefLocalSpheresFkCG in rainbow_ik_cg.hh) against the
        //      environment -- this runs unconditionally, so a hand still gets checked even
        //      when the caller hasn't attached anything to it;
        //   2. IF the environment has attachments (e.g. a held object) registered for these
        //      end effectors, those too, against the environment and against each other
        //      (attachment_environment_collision / attachment_attachment_collision are
        //      no-ops when `environment.attachments` is empty, so this is safe either way).
        // Does NOT check either of the above against the robot's own body (torso/arms/base)
        // -- that requires a resolved ambient `q` (see resolve_block + Ambient::fkcc_attach
        // below), which this deliberately skips to stay IK-free. Intended to prune
        // obviously-bad samples before paying for resolve_block(); a `false` return here is
        // not a full validity guarantee.
        // Assumes end effector 0 is the left hand and 1 is the right hand (this robot's
        // configured order -- see GenOptions::end_effectors), matching t_mid_left/t_mid_right.
        template <std::size_t rake>
        static inline auto eefs_collision_free(
            const vamp::collision::Environment<FloatVector<rake>> &environment,
            const StateBlock<rake> &x_in) noexcept -> bool
        {
            using V = FloatVector<rake, 1>;

            auto world_poses = eef_world_poses<rake>(x_in);

            //
            // 1. each hand's own geometry vs. environment (no attachment needed)
            //
            {% if param_eef_spheres_code_vars > 0 %}std::array<V, {{param_eef_spheres_code_vars}}> v;{% endif %}
            std::array<V, {{param_eef_spheres_code_output}}> y;
            const auto &x = world_poses;

            {{param_eef_spheres_code}}

            {% for i in range(n_left_eef_spheres + n_right_eef_spheres) %}
            if (sphere_environment_in_collision(environment, y[{{i}} * 4 + 0], y[{{i}} * 4 + 1], y[{{i}} * 4 + 2], y[{{i}} * 4 + 3]))
            {
                return false;
            }
            {% endfor %}

            //
            // 2. attachments (if any) vs. environment / vs. each other
            //
            set_attachment_pose(environment, 0, to_isometry(&world_poses[0]));
            set_attachment_pose(environment, 1, to_isometry(&world_poses[12]));

            if (attachment_environment_collision(environment))
            {
                return false;
            }

            if (attachment_attachment_collision(environment))
            {
                return false;
            }

            return true;
        }
        {% endif %}

        // Batched task-space -> ambient-configuration resolve, for the FK/collision-checking
        // boundary; the counterpart of a future ParameterizedLocalPlanner's per-lane IK solve.
        // `x` decomposes exactly as trace_rby1_constrained_sample's State layout: base(4) +
        // torso(6) + psi_left(1) + psi_right(1) + t_mid_pose(7). param_ik_code additionally
        // reads left_gcp/right_gcp/t_mid_left/t_mid_right directly by name (see those members
        // above) rather than taking them from `x`. Returns {false, _} if either arm's
        // reach_violation is nonzero (RainbowArmParamResult: no valid IK solution on this GCP
        // branch for the requested pose -- see rainbow_arm_parameterization.hh) or the
        // resolved ambient configuration falls outside the robot's own joint limits.
        template <std::size_t rake>
        static inline auto resolve_block(const StateBlock<rake> &x) noexcept
            -> std::pair<bool, Ambient::ConfigurationBlock<rake>>
        {
            using V = FloatVector<rake, 1>;

            std::array<V, 4> base{x[0], x[1], x[2], x[3]};
            std::array<V, 6> torso{x[4], x[5], x[6], x[7], x[8], x[9]};
            const auto psi_left = x[10];
            const auto psi_right = x[11];
            std::array<V, 7> t_mid_pose{x[12], x[13], x[14], x[15], x[16], x[17], x[18]};

            FloatVector<rake, {{param_ik_code_vars}}> v;
            Ambient::ConfigurationBlock<rake> q;
            FloatVector<rake, {{param_ik_num_unclipped}}> u_left;
            FloatVector<rake, 1> reach_violation_left;
            FloatVector<rake, 1> loss_left;
            FloatVector<rake, {{param_ik_num_unclipped}}> u_right;
            FloatVector<rake, 1> reach_violation_right;
            FloatVector<rake, 1> loss_right;

            {{param_ik_code}}

            if ((reach_violation_left[0] > 0.0f).any() or (reach_violation_right[0] > 0.0f).any())
            {
                return {false, q};
            }

            {% for i in range(n_q) %}
            if ((q[{{i}}] < V(Ambient::lower_bound[{{i}}])).any() or (q[{{i}}] > V(Ambient::upper_bound[{{i}}])).any())
            {
                return {false, q};
            }
            {% endfor %}

            return {true, q};
        }
    };
    {% endif %}
};
}

// NOLINTEND(*-magic-numbers)
