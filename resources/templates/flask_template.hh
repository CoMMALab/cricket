#pragma once

#include <vamp/vector.hh>
#include <vamp/vector/math.hh>
#include <vamp/collision/environment.hh>
#include <vamp/collision/validity.hh>
#include <vamp/planning/flask.hh>
#include <vamp/planning/nn.hh>

#include <Eigen/Geometry>
#include <nigh/so3_space.hpp>
#include <nigh/cartesian_space.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

// clang-format off
// NOLINTBEGIN(*-magic-numbers)
namespace vamp::robots
{
struct {{name}}
{
    static constexpr const char *name = "{{module_name}}";
    static constexpr std::size_t dimension = {{n_z}};
    static constexpr std::size_t sample_dimension = {{n_z}};
    static constexpr std::size_t flat_dimension = {{n_q}};
    static constexpr std::size_t n_spheres = {{n_spheres}};
    static constexpr float min_radius = {{min_radius}};
    static constexpr float max_radius = {{max_radius}};
    static constexpr std::size_t resolution = {{resolution}};
    static constexpr bool euclidean = false;
    static constexpr bool flask = true;
    static constexpr std::array<std::size_t, 0> so3_offsets = {};
    static inline float rho = {{rho}};

    static constexpr std::array<std::string_view, dimension> joint_names = {"{{join(z_joint_names, "\", \"")}}"};
    static constexpr const char *end_effector = "{{end_effector}}";

    static constexpr std::array<float, flat_dimension> velocity_limits = {
        {{join(velocity_limits, ", ")}}
    };

    static constexpr std::array<float, flat_dimension> effort_limits = {
        {{join(effort_limits, ", ")}}
    };

    using Configuration = FloatVector<dimension>;
    struct alignas(FloatVectorAlignment) ConfigurationArray
        : std::array<FloatT, dimension>
    {
    };
    using Sample = FloatVector<sample_dimension>;

    using NNKey = std::tuple<vamp::planning::NNFloatArray<dimension>>;

    using NNSpace = unc::robotics::nigh::metric::CartesianSpace<
        unc::robotics::nigh::metric::Space<vamp::planning::NNFloatArray<dimension>, unc::robotics::nigh::metric::LP<2>>>;

    static inline auto nn_key(float *cfg_ptr) noexcept -> NNKey
    {
        return NNKey{vamp::planning::NNFloatArray<dimension>{cfg_ptr}};
    }

    struct alignas(FloatVectorAlignment) ConfigurationBuffer
        : std::array<float, Configuration::num_scalars_rounded>
    {
    };

    template <std::size_t rake>
    using ConfigurationBlock = FloatVector<rake, {{n_x}}>;

    template <std::size_t rake>
    struct Spheres
    {
        FloatVector<rake, n_spheres> x;
        FloatVector<rake, n_spheres> y;
        FloatVector<rake, n_spheres> z;
        FloatVector<rake, n_spheres> r;
    };

    alignas(Configuration::S::Alignment) static constexpr std::array<float, dimension> s_m{
        {{join(z_range, ", ")}}
    };

    alignas(Configuration::S::Alignment) static constexpr std::array<float, dimension> s_a{
        {{join(z_lower, ", ")}}
    };

    alignas(Configuration::S::Alignment) static constexpr std::array<float, dimension> d_m{
        {{join(z_descale, ", ")}}
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
        {% for index in range(n_z) -%}
        q[{{index}}] = {{ at(z_lower, index) }} + (q[{{index}}] * {{ at(z_range, index) }});
        {%- endfor %}
    }

    template <std::size_t rake>
    static inline void descale_configuration_block(ConfigurationBlock<rake> & q) noexcept
    {
        {% for index in range(n_z) -%}
        q[{{index}}] = {{ at(z_descale, index) }} * (q[{{index}}] - {{ at(z_lower, index) }});
        {%- endfor %}
    }

    inline static auto space_measure() noexcept -> float
    {
        return {{z_measure}};
    }

    alignas(Configuration::S::Alignment) static constexpr std::array<float, dimension> lower_bound{
        {{join(z_lower, ", ")}}
    };

    alignas(Configuration::S::Alignment) static constexpr std::array<float, dimension> upper_bound{
        {{join(z_upper, ", ")}}
    };

    static inline auto in_bounds(const Configuration &x) -> bool
    {
        return (x <= Configuration(upper_bound)).all() and (x >= Configuration(lower_bound)).all();
    }

    static inline auto sample(const Sample &x_in) -> Configuration
    {
        Configuration q = x_in;
        scale_configuration(q);
        return q;
    }

    static inline auto distance(const Configuration &a_in, const Configuration &b_in) -> float
    {
        return a_in.distance(b_in);
    }

    static inline auto optimal_time(const Configuration &a_in, const Configuration &b_in) noexcept -> float
    {
        return vamp::planning::flask::optimal_time<{{name}}>(a_in, b_in);
    }

    static inline auto cost(const Configuration &a_in, const Configuration &b_in) noexcept -> float
    {
        return vamp::planning::flask::cost<{{name}}>(a_in, b_in);
    }

    static inline auto cost_grad(const Configuration &a_in, const Configuration &b_in) noexcept
        -> vamp::planning::flask::LQMTCostGrad<dimension>
    {
        return vamp::planning::flask::cost_grad<{{name}}>(a_in, b_in);
    }

    static inline auto eval(const Configuration &a_in, const Configuration &b_in, float T, float t) noexcept
        -> std::array<float, {{n_x}}>
    {
        {% if flask_interpolate_code_vars > 0 %}std::array<float, {{flask_interpolate_code_vars}}> v;{% endif %}
        std::array<float, {{n_x}}> y;
        const auto a = a_in.to_array();
        const auto b = b_in.to_array();
        {{flask_interpolate_code}}
        return y;
    }

    static inline auto torques(const std::array<float, {{n_x}}> &x) noexcept -> std::array<float, flat_dimension>
    {
        {% if flask_rnea_code_vars > 0 %}std::array<float, {{flask_rnea_code_vars}}> v;{% endif %}
        std::array<float, flat_dimension> y;
        {{flask_rnea_code}}
        return y;
    }

    static inline auto interpolate(const Configuration &a_in, const Configuration &b_in, float t) -> Configuration
    {
        {% if flask_interpolate_code_vars > 0 %}std::array<float, {{flask_interpolate_code_vars}}> v;{% endif %}
        alignas(FloatVectorAlignment)
            std::array<float, std::max<std::size_t>({{n_x}}, Configuration::num_scalars_rounded)> y;
        const auto a = a_in.to_array();
        const auto b = b_in.to_array();
        const float T = optimal_time(a_in, b_in);
        {{flask_interpolate_code}}
        return Configuration(y.data());
    }

    template <std::size_t rake>
    static inline void interpolate_block(
        const Configuration &a,
        const Configuration &b,
        const FloatVector<rake> &t,
        ConfigurationBlock<rake> &out) noexcept
    {
        using V = FloatVector<rake, 1>;
        const V T = V::fill(optimal_time(a, b));
        {% if flask_interpolate_block_code_vars > 0 %}std::array<V, {{flask_interpolate_block_code_vars}}> v;{% endif %}
        {{flask_interpolate_block_code}}
    }

    template <std::size_t rake>
    static inline bool limits_check(const ConfigurationBlock<rake> &x) noexcept
    {
        {% for index in range(n_z) -%}
        if (not ((x[{{index}}] >= {{ at(z_lower, index) }}).all() and (x[{{index}}] <= {{ at(z_upper, index) }}).all()))
        {
            return false;
        }
        {% endfor %}

        {% if flask_rnea_block_code_vars > 0 %}std::array<FloatVector<rake, 1>, {{flask_rnea_block_code_vars}}> v;{% endif %}
        std::array<FloatVector<rake, 1>, flat_dimension> tau;
        {{flask_rnea_block_code}}

        {% for index in range(n_q) -%}
        if ((tau[{{index}}].abs() > {{ at(effort_limits, index) }}).any())
        {
            return false;
        }
        {% endfor %}

        return true;
    }

    template <std::size_t rake>
    static inline void sphere_fk(const ConfigurationBlock<rake> &x, Spheres<rake> &out) noexcept
    {
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

    template <std::size_t rake>
        static inline auto fkcc_debug(
            const vamp::collision::Environment<FloatVector<rake>> &environment,
            const ConfigurationBlock<rake> &x) noexcept -> Debug
    {
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

    template <std::size_t rake>
        static inline bool fkcc(
            const vamp::collision::Environment<FloatVector<rake>> &environment,
            const ConfigurationBlock<rake> &x) noexcept
    {
        if (not limits_check<rake>(x))
        {
            return false;
        }

        std::array<FloatVector<rake, 1>, {{ccfk_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{ccfk_code_output}}> y;

        {{ccfk_code}}
        {% include "ccfk" %}

        return true;
    }

    template <std::size_t rake>
    static inline bool fkcc_attach(
        const vamp::collision::Environment<FloatVector<rake>> &environment,
        const ConfigurationBlock<rake> &x) noexcept
    {
        if (not limits_check<rake>(x))
        {
            return false;
        }

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

    static inline auto eefk(const std::array<float, dimension> &x) noexcept -> Eigen::Isometry3f
    {
        std::array<float, {{eefk_code_vars}}> v;
        std::array<float, {{eefk_code_output}}> y;

        {{eefk_code}}

        return to_isometry(y.data());
    }

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
