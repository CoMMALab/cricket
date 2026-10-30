#pragma once

#include <vamp/vector.hh>
#include <vamp/vector/math.hh>
#include <vamp/collision/environment.hh>
#include <vamp/collision/validity.hh>

#include <Eigen/Geometry>

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
    static constexpr float min_radius_mobile = {{min_radius_mobile}};
    static constexpr float max_radius_mobile = {{max_radius_mobile}};
    static constexpr float min_bounding_radius_mobile = {{min_bounding_radius_mobile}};
    static constexpr float max_bounding_radius_mobile = {{max_bounding_radius_mobile}};
    static constexpr std::array<float, 3> base_position = { {{join(base_position, ", ")}} };
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

    // The robot is its own Space for the planners: a Space names its points State, and here they are Configurations
    // (and its own ambient robot). Planners and constraints templated on <Robot, Space = Robot> use these names, so
    // spaces with another State (e.g. a configuration with extra coordinates) plug in without the planners
    // assuming a Configuration. Nothing is generated for such spaces: they are header-only templates in vamp.
    using Ambient = {{name}};
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
    // that many rows.
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
        // V is referenced by the SIMD blends emitted for the se(3) log map's conditionals
        using V = FloatVector<rake, 1>;
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

    {% if has_bounded_ls %}
    // Bounded least-squares projection step for the TSR error.
    // Input: the solver input above, then lower - q (dimension entries) and upper - q (dimension entries): the
    // allowed step to each joint limit. Minimises the damped least squares subject to those bounds.
    template <std::size_t rake, typename InputVector>
    static inline auto solve_tsr_error_bounded_ls(const InputVector &x, ConfigurationBlock<rake> &y) noexcept
    {
        // V is referenced by the SIMD blends emitted for the solver's conditionals
        using V = FloatVector<rake, 1>;
        std::array<V, {{solve_tsr_error_bounded_ls_code_vars}}> v;

        {{solve_tsr_error_bounded_ls_code}}
    }
    {% endif %}

    {% if num_end_effectors > 1 %}
    // Relative-pose TSR between the first two end-effectors.
    // Input: q (dimension), then the reference relative transform lTr (7), lb (6), ub (6).
    // Output: d(err)/dq (6 * dimension, row-major), then the raw error (6).
    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto tsr_bimanual_error(const InputVector &x, OutputVector &out) noexcept
    {
        // V is referenced by the SIMD blends emitted for the se(3) log map's conditionals
        using V = FloatVector<rake, 1>;
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

    {% if has_bounded_ls %}
    // Bounded least-squares projection step for the relative-pose TSR error.
    // Input: the solver input above, then lower - q (dimension entries) and upper - q (dimension entries): the
    // allowed step to each joint limit. Minimises the damped least squares subject to those bounds.
    template <std::size_t rake, typename InputVector>
    static inline auto solve_tsr_relative_error_bounded_ls(const InputVector &x, ConfigurationBlock<rake> &y) noexcept
    {
        // V is referenced by the SIMD blends emitted for the solver's conditionals
        using V = FloatVector<rake, 1>;
        std::array<V, {{solve_tsr_relative_error_bounded_ls_code_vars}}> v;

        {{solve_tsr_relative_error_bounded_ls_code}}
    }
    {% endif %}
    {% endif %}
    {% endif %}

    {% if has_spline_constraint %}
    //
    // Spline constraint: the first end-effector's position lies on a cubic B-spline curve. The curve
    // parameter t in [0, 1] is an extra unknown next to the joints, so the projection works on
    // (q, t) and the error is the 3-vector  FK(q).translation - P(t).
    //

    // The spline has this many control points and knots (fixed when the kernel is generated)
    static constexpr std::size_t spline_control_points = {{spline_control_points}};
    static constexpr std::size_t spline_knots = {{spline_knots}};

    // Unknowns of the spline projection: the joints, then t
    static constexpr std::size_t spline_variables = dimension + 1;

    // Input: q (dimension), then t, then the control points (3 * spline_control_points, x y z per point), then the
    // knots (spline_knots). Output: d(err)/d(q, t) (3 * spline_variables, row-major, t the last column), then
    // the raw error (3).
    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto spline_error(const InputVector &x, OutputVector &out) noexcept
    {
        // V is referenced by the SIMD blends emitted for the span selection's conditionals
        using V = FloatVector<rake, 1>;
        std::array<V, {{spline_error_code_vars}}> v;
        std::array<V, {{spline_error_code_output}}> y;

        {{spline_error_code}}

        for (auto i = 0U; i < {{spline_error_code_output}}; ++i)
        {
            out[i] = y[i];
        }
    }

    // Projection steps for the spline error, as for the TSR error. Input: the outputs of spline_error (J, then
    // err). Output: the gradient (spline_variables entries: the joints, then t), applied as x <- x - gradient. These
    // do not bound t, so the caller clamps it to [0, 1]; the bounded variant below does.
    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto solve_spline_error_lm_inner(const InputVector &x, OutputVector &y) noexcept
    {
        std::array<FloatVector<rake, 1>, {{solve_spline_error_lm_inner_code_vars}}> v;

        {{solve_spline_error_lm_inner_code}}
    }

    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto solve_spline_error_lm_outer(const InputVector &x, OutputVector &y) noexcept
    {
        std::array<FloatVector<rake, 1>, {{solve_spline_error_lm_outer_code_vars}}> v;

        {{solve_spline_error_lm_outer_code}}
    }

    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto solve_spline_error_gradient_descent(const InputVector &x, OutputVector &y) noexcept
    {
        std::array<FloatVector<rake, 1>, {{solve_spline_error_gradient_descent_code_vars}}> v;

        {{solve_spline_error_gradient_descent_code}}
    }

    // Bounded least-squares projection step for the spline error.
    // Input: the outputs of spline_error (J then err), then lower - x (spline_variables entries) and upper - x
    // (spline_variables entries): the allowed step to each bound, where x is q followed by t. The joints' entries
    // are their limits minus q; the last entry of each is 0 - t and 1 - t, which keeps t in [0, 1].
    // Output: the gradient (spline_variables entries, so one more than a ConfigurationBlock holds), applied as
    // x <- x - gradient; its last entry is the step in t.
    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto solve_spline_error_bounded_ls(const InputVector &x, OutputVector &y) noexcept
    {
        // V is referenced by the SIMD blends emitted for the solver's conditionals
        using V = FloatVector<rake, 1>;
        std::array<V, {{solve_spline_error_bounded_ls_code_vars}}> v;

        {{solve_spline_error_bounded_ls_code}}
    }

    // Starting guess for t: the curve parameter between the waypoint nearest to the end-effector at q and that waypoint's
    // nearer neighbour, weighted by the distances to the two.
    // Input: q (dimension), then the waypoints the spline was fitted through (3 * spline_control_points, x y z per
    // waypoint), then their curve parameters (spline_control_points). Output: t (1).
    //
    // NOTE: still a simple seed, exact only on the segment between the two waypoints. A better one, for faster
    // convergence of whatever refines it, would project the end-effector onto the two chords next to the nearest
    // waypoint and interpolate t along the closer one.
    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto spline_seed(const InputVector &x, OutputVector &out) noexcept
    {
        // V is referenced by the SIMD blends emitted for the nearest-waypoint search
        using V = FloatVector<rake, 1>;
        std::array<V, {{spline_seed_code_vars}}> v;
        std::array<V, {{spline_seed_code_output}}> y;

        {{spline_seed_code}}

        for (auto i = 0U; i < {{spline_seed_code_output}}; ++i)
        {
            out[i] = y[i];
        }
    }

    // The closest point of the spline to the end-effector, as a function of q alone.
    // Input: q (dimension), then the control points (3 * spline_control_points, x y z per point), the knots
    // (spline_knots), the waypoints the spline was fitted through (3 * spline_control_points) and their curve
    // parameters (spline_control_points). Output: [t*, squared distance |FK(q) - P(t*)|^2] (2).
    //
    // t* is the seed (the distance-weighted parameter between the nearest waypoint and its nearer neighbour) refined by a fixed number of
    // Gauss-Newton steps on t with q fixed. A squared distance near zero proves q is on the curve; a larger one only
    // says no nearby point was found (a seed on the wrong branch gives the distance to that branch).
    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto spline_closest_t(const InputVector &x, OutputVector &out) noexcept
    {
        // V is referenced by the SIMD blends emitted for the seed search and the step limits
        using V = FloatVector<rake, 1>;
        std::array<V, {{spline_closest_t_code_vars}}> v;
        std::array<V, {{spline_closest_t_code_output}}> y;

        {{spline_closest_t_code}}

        for (auto i = 0U; i < {{spline_closest_t_code_output}}; ++i)
        {
            out[i] = y[i];
        }
    }

    //
    // The same error and projection steps with t given, for planning in SplineSpace below, where t is part of the
    // state and a projection solves for the joints alone.
    //

    // Input: as spline_error (q, t, control points, knots). Output: d(err)/dq (3 * dimension, row-major, no t
    // column), then the raw error (3).
    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto spline_error_fixed_t(const InputVector &x, OutputVector &out) noexcept
    {
        // V is referenced by the SIMD blends emitted for the span selection's conditionals
        using V = FloatVector<rake, 1>;
        std::array<V, {{spline_error_fixed_t_code_vars}}> v;
        std::array<V, {{spline_error_fixed_t_code_output}}> y;

        {{spline_error_fixed_t_code}}

        for (auto i = 0U; i < {{spline_error_fixed_t_code_output}}; ++i)
        {
            out[i] = y[i];
        }
    }

    // Projection steps for spline_error_fixed_t. Input: its outputs (J, then err), and for the bounded variant then
    // lower - q and upper - q (dimension entries each). Output: the gradient (dimension entries), applied as
    // q <- q - gradient.
    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto solve_spline_fixed_t_lm_inner(const InputVector &x, OutputVector &y) noexcept
    {
        std::array<FloatVector<rake, 1>, {{solve_spline_fixed_t_lm_inner_code_vars}}> v;

        {{solve_spline_fixed_t_lm_inner_code}}
    }

    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto solve_spline_fixed_t_lm_outer(const InputVector &x, OutputVector &y) noexcept
    {
        std::array<FloatVector<rake, 1>, {{solve_spline_fixed_t_lm_outer_code_vars}}> v;

        {{solve_spline_fixed_t_lm_outer_code}}
    }

    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto solve_spline_fixed_t_gradient_descent(const InputVector &x, OutputVector &y) noexcept
    {
        std::array<FloatVector<rake, 1>, {{solve_spline_fixed_t_gradient_descent_code_vars}}> v;

        {{solve_spline_fixed_t_gradient_descent_code}}
    }

    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto solve_spline_fixed_t_bounded_ls(const InputVector &x, OutputVector &y) noexcept
    {
        // V is referenced by the SIMD blends emitted for the solver's conditionals
        using V = FloatVector<rake, 1>;
        std::array<V, {{solve_spline_fixed_t_bounded_ls_code_vars}}> v;

        {{solve_spline_fixed_t_bounded_ls_code}}
    }

    //
    // SplineSpace: the space (q, t) of this robot's configurations extended with the spline parameter t in [0, 1],
    // to plan over instead of the robot's own Configuration when the end-effector has to follow a curve. A state is
    // the joints followed by t; on the manifold the end-effector sits at P(t), so t says where on the curve a state
    // is. Collision checking and kinematics see the ambient configuration (to_ambient); the distance, which the
    // nearest-neighbor tree shares by reading the first `dimension` scalars of a state buffer, ignores t. Exposes the
    // same State/StateArray/StateBlock<rake>/StateBuffer family as the outer struct, so vamp's planners are templated
    // on either.
    //
    struct SplineSpace
    {
        static constexpr const char *name = "spline_space";

        // The robot this space sits over: what collision checking and the spline kernels above work on
        using Ambient = {{name}};

        // The state is the joints, then t
        static constexpr std::size_t t_index = {{name}}::dimension;
        static constexpr std::size_t dimension = {{name}}::dimension + 1;
        static constexpr std::size_t sample_dimension = {{name}}::sample_dimension;

        using State = FloatVector<dimension>;

        struct alignas(FloatVectorAlignment) StateArray
            : std::array<FloatT, dimension>
        {
        };

        struct alignas(FloatVectorAlignment) StateBuffer
            : std::array<float, State::num_scalars_rounded>
        {
        };

        template <std::size_t rake>
        using StateBlock = FloatVector<rake, dimension>;

        static inline auto make_state(const Configuration &q, float t) noexcept -> State
        {
            std::array<float, dimension> state;
            const auto array = q.to_array();
            for (auto j = 0U; j < {{name}}::dimension; ++j)
            {
                state[j] = array[j];
            }

            state[t_index] = t;
            return State(state);
        }

        static inline auto to_ambient(const State &s) noexcept -> Configuration
        {
            std::array<float, {{name}}::dimension> q;
            const auto array = s.to_array();
            for (auto j = 0U; j < {{name}}::dimension; ++j)
            {
                q[j] = array[j];
            }

            return Configuration(q);
        }

        static inline auto parameter(const State &s) noexcept -> float
        {
            return s.to_array()[t_index];
        }

        static inline auto with_ambient(const State &s, const Configuration &q) noexcept -> State
        {
            return make_state(q, parameter(s));
        }

        // Where a goal-directed step takes t: `step` further along the curve, up to its end
        static inline auto advance_parameter(float t, float step) noexcept -> float
        {
            return (t + step < 1.0f) ? t + step : 1.0f;
        }

        // The metric ignores t
        static inline auto distance(const State &a, const State &b) noexcept -> float
        {
            return {{name}}::distance(to_ambient(a), to_ambient(b));
        }

        static inline auto interpolate(const State &a, const State &b, float f) noexcept -> State
        {
            const auto ta = parameter(a);
            const auto tb = parameter(b);
            return make_state({{name}}::interpolate(to_ambient(a), to_ambient(b), f), ta + (tb - ta) * f);
        }

        static inline auto in_bounds(const State &s) noexcept -> bool
        {
            const auto t = parameter(s);
            return {{name}}::in_bounds(to_ambient(s)) and t >= 0.0f and t <= 1.0f;
        }
    };
    {% endif %}

    {% if has_position_target_constraint %}
    //
    // Position target constraint: the first end-effector's position equals a target point that the caller supplies, for
    // planning along a curve in SplineSpace where the runtime evaluates the curve at the parameter t itself. The
    // projection solves for the joints alone, with the fixed-parameter solvers solve_spline_fixed_t_* of the spline
    // constraint.
    //

    static constexpr bool has_position_target = true;

    // Input: q (dimension), then the target point (3). Output: d(err)/dq (3 * dimension, row-major), then the raw error
    // (3): the layout of spline_error_fixed_t.
    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto position_target_error(const InputVector &x, OutputVector &out) noexcept
    {
        std::array<FloatVector<rake, 1>, {{position_target_error_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{position_target_error_code_output}}> y;

        {{position_target_error_code}}

        for (auto i = 0U; i < {{position_target_error_code_output}}; ++i)
        {
            out[i] = y[i];
        }
    }
    {% endif %}

    {% if has_pose_slide_constraint %}
    //
    // Pose curve with a variable parameter: the projection steps for the TSR error (tsr_error, 6 rows) of a pose curve
    // whose parameter t is an unknown of the projection next to the joints (planning in SplineSpace with the
    // end-effector sliding along its segment).
    //

    static constexpr bool has_pose_slide = true;

    // Input: J (6 * (dimension + 1), row-major: d(err)/dq from tsr_error with a column d(err)/dt appended by the
    // caller), then err (6). Output: the gradient (dimension + 1 entries: the joints, then t), applied as
    // x <- x - gradient. These do not bound t: the caller clamps it to its segment; the bounded variant enforces it.
    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto solve_pose_slide_error_lm_inner(const InputVector &x, OutputVector &y) noexcept
    {
        std::array<FloatVector<rake, 1>, {{solve_pose_slide_error_lm_inner_code_vars}}> v;

        {{solve_pose_slide_error_lm_inner_code}}
    }

    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto solve_pose_slide_error_lm_outer(const InputVector &x, OutputVector &y) noexcept
    {
        std::array<FloatVector<rake, 1>, {{solve_pose_slide_error_lm_outer_code_vars}}> v;

        {{solve_pose_slide_error_lm_outer_code}}
    }

    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto solve_pose_slide_error_gradient_descent(const InputVector &x, OutputVector &y) noexcept
    {
        std::array<FloatVector<rake, 1>, {{solve_pose_slide_error_gradient_descent_code_vars}}> v;

        {{solve_pose_slide_error_gradient_descent_code}}
    }

    // Input: as above, then lower - x and upper - x (dimension + 1 entries each): the allowed step to each bound,
    // where x is q followed by t. The last entry of each is the step allowed to the segment's ends.
    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto solve_pose_slide_error_bounded_ls(const InputVector &x, OutputVector &y) noexcept
    {
        // V is referenced by the SIMD blends emitted for the solver's conditionals
        using V = FloatVector<rake, 1>;
        std::array<V, {{solve_pose_slide_error_bounded_ls_code_vars}}> v;

        {{solve_pose_slide_error_bounded_ls_code}}
    }
    {% endif %}

    {% if has_catmull_rom_constraint %}
    //
    // Catmull-Rom constraint: the first end-effector's position lies on a Catmull-Rom curve through
    // catmull_rom_points points, at a given curve parameter t in [0, 1] (planning in SplineSpace). The projection
    // solves for the joints alone, with the fixed-parameter solvers solve_spline_fixed_t_* of the spline constraint.
    //

    // The curve passes through this many points; its data is extended by a phantom point on each side
    static constexpr std::size_t catmull_rom_points = {{catmull_rom_points}};

    // Input: q (dimension), then t, then the extended points (3 * (catmull_rom_points + 2), x y z per point), then the
    // extended knots (catmull_rom_points + 2). Output: d(err)/dq (3 * dimension, row-major), then the raw error (3):
    // the layout of spline_error_fixed_t.
    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto catmull_rom_error(const InputVector &x, OutputVector &out) noexcept
    {
        // V is referenced by the SIMD blends emitted for the segment selection's conditionals
        using V = FloatVector<rake, 1>;
        std::array<V, {{catmull_rom_error_code_vars}}> v;
        std::array<V, {{catmull_rom_error_code_output}}> y;

        {{catmull_rom_error_code}}

        for (auto i = 0U; i < {{catmull_rom_error_code_output}}; ++i)
        {
            out[i] = y[i];
        }
    }
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

    {% if has_bounded_ls %}
    // Bounded least-squares projection step for the center-of-mass error.
    // Input: the solver input above, then lower - q (dimension entries) and upper - q (dimension entries): the
    // allowed step to each joint limit. Minimises the damped least squares subject to those bounds.
    template <std::size_t rake, typename InputVector>
    static inline auto solve_com_error_bounded_ls(const InputVector &x, ConfigurationBlock<rake> &y) noexcept
    {
        // V is referenced by the SIMD blends emitted for the solver's conditionals
        using V = FloatVector<rake, 1>;
        std::array<V, {{solve_com_error_bounded_ls_code_vars}}> v;

        {{solve_com_error_bounded_ls_code}}
    }
    {% endif %}
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

    {% if has_bounded_ls %}
    // Bounded least-squares projection step for the closed-loop error.
    // Input: the solver input above, then lower - q (dimension entries) and upper - q (dimension entries): the
    // allowed step to each joint limit. Minimises the damped least squares subject to those bounds.
    template <std::size_t rake, typename InputVector>
    static inline auto solve_closed_loop_error_bounded_ls(const InputVector &x, ConfigurationBlock<rake> &y) noexcept
    {
        // V is referenced by the SIMD blends emitted for the solver's conditionals
        using V = FloatVector<rake, 1>;
        std::array<V, {{solve_closed_loop_error_bounded_ls_code_vars}}> v;

        {{solve_closed_loop_error_bounded_ls_code}}
    }
    {% endif %}
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
        // V is referenced by the SIMD blends emitted for the se(3) log map's conditionals
        using V = FloatVector<rake, 1>;
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

    {% if has_bounded_ls %}
    // Bounded least-squares projection step for the lead-screw error.
    // Input: the solver input above, then lower - q (dimension entries) and upper - q (dimension entries): the
    // allowed step to each joint limit. Minimises the damped least squares subject to those bounds.
    template <std::size_t rake, typename InputVector>
    static inline auto solve_lead_screw_error_bounded_ls(const InputVector &x, ConfigurationBlock<rake> &y) noexcept
    {
        // V is referenced by the SIMD blends emitted for the solver's conditionals
        using V = FloatVector<rake, 1>;
        std::array<V, {{solve_lead_screw_error_bounded_ls_code_vars}}> v;

        {{solve_lead_screw_error_bounded_ls_code}}
    }
    {% endif %}
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

};
}

// NOLINTEND(*-magic-numbers)
