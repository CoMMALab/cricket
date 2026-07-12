#pragma once

#include <vamp/vector.hh>
#include <vamp/vector/math.hh>
#include <vamp/collision/environment.hh>
#include <vamp/collision/validity.hh>
#include <vamp/planning/nn.hh>
{% if has_flask %}#include <vamp/planning/flask.hh>{% endif %}

#include <Eigen/Geometry>
#include <nigh/so3_space.hpp>
#include <nigh/cartesian_space.hpp>
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

    using NNKey = std::tuple<
        {% for seg in nn_segments -%}
        {% if seg.type == "LP" %}vamp::planning::NNFloatArray<{{seg.size}}>{% else %}Eigen::Quaternion<float>{% endif %}{% if not loop.is_last %},{% endif %}
        {% endfor %}>;

    using NNSpace = unc::robotics::nigh::metric::CartesianSpace<
        {% for seg in nn_segments -%}
        {% if seg.type == "LP" %}unc::robotics::nigh::metric::Space<vamp::planning::NNFloatArray<{{seg.size}}>, unc::robotics::nigh::metric::LP<2>>{% else %}unc::robotics::nigh::metric::Space<Eigen::Quaternion<float>, unc::robotics::nigh::metric::SO3>{% endif %}{% if not loop.is_last %},{% endif %}
        {% endfor %}>;

    static inline auto nn_key(float *cfg_ptr) noexcept -> NNKey
    {
        return NNKey{
            {% for seg in nn_segments -%}
            {% if seg.type == "LP" %}vamp::planning::NNFloatArray<{{seg.size}}>{cfg_ptr + {{seg.offset}}}{% else %}Eigen::Quaternion<float>(cfg_ptr[{{seg.offset + 3}}], cfg_ptr[{{seg.offset}}], cfg_ptr[{{seg.offset + 1}}], cfg_ptr[{{seg.offset + 2}}]){% endif %}{% if not loop.is_last %},{% endif %}
            {% endfor %}};
    }

    struct alignas(FloatVectorAlignment) ConfigurationBuffer
        : std::array<float, Configuration::num_scalars_rounded>
    {
    };

    template <std::size_t rake>
    using ConfigurationBlock = FloatVector<rake, dimension>;

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

        {% for i in range(n_spheres) %}
        output.first.emplace_back(
            sphere_environment_get_collisions<decltype(x[0])>(
                environment,
                y[{{ i * 4 + 0 }}],
                y[{{ i * 4 + 1 }}],
                y[{{ i * 4 + 2 }}],
                y[{{ i * 4 + 3 }}]));
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

        // attaching at {{ end_effector }}
        set_attachment_pose(environment, to_isometry(&y[{{ccfkee_code_output - 12}}]));

        //
        // attachment vs. environment collisions
        //
        if (attachment_environment_collision(environment)) [[unlikely]]
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
        if (attachment_sphere_collision<decltype(x[0])>(environment,
                                                        y[{{(n_spheres + link_bs) * 4 + 0}}],
                                                        y[{{(n_spheres + link_bs) * 4 + 1}}],
                                                        y[{{(n_spheres + link_bs) * 4 + 2}}],
                                                        y[{{(n_spheres + link_bs) * 4 + 3}}])) [[unlikely]]
        {
            {% for j in range(length(link_spheres)) %}
            {% set sphere_index = at(link_spheres, j) %}
            if (attachment_sphere_collision<decltype(x[0])>(environment,
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

        return true;
    }

    static inline auto eefk(const std::array<float, {{n_q}}> &x) noexcept -> Eigen::Isometry3f
    {
        std::array<float, {{eefk_code_vars}}> v;
        std::array<float, {{eefk_code_output}}> y;

        {{eefk_code}}

        return to_isometry(y.data());
    }

    {% if has_constraints %}
    //
    // TSR (task-space region) constraint functions
    //

    static constexpr std::size_t n_eef = {{num_end_effectors}};
    static constexpr std::array<std::string_view, {{num_end_effectors}}> end_effectors = {"{{join(end_effectors, "\", \"")}}"};

    // Input: q (dimension), then per end-effector rTe (7), wTr (7), lb (6), ub (6); transforms
    // are wxyz quaternion + xyz translation. Output: d(err)/dq (6 * n_eef * dimension,
    // row-major), then the raw un-hinged error (6 * n_eef).
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
    // Output: d(err)/dq (6 * dimension, row-major), then the raw un-hinged error (6).
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
};
}

// NOLINTEND(*-magic-numbers)
