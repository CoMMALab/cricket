#pragma once

#include <vamp/vector.hh>
#include <vamp/vector/math.hh>
#include <vamp/collision/environment.hh>
#include <vamp/collision/validity.hh>
#include <vamp/planning/nn.hh>

#include <Eigen/Geometry>
#include <nigh/so3_space.hpp>
#include <nigh/cartesian_space.hpp>
// NOLINTBEGIN(*-magic-numbers)
namespace vamp::robots
{
struct {{name}}
{
    static constexpr char* name = "{{lower(name)}}";
    static constexpr std::size_t dimension = {{7 + 1}};
    static constexpr std::size_t sample_dimension = {{6 + 1}};
    
    static constexpr std::size_t n_spheres = {{n_spheres}};
    static constexpr float min_radius = {{min_radius}};
    static constexpr float max_radius = {{max_radius}};
    static constexpr std::size_t resolution = {{resolution}};

    static constexpr std::size_t ambient_dimension = {{ambient_nq}};

    static constexpr bool use_parameterized_ik = true;

    // Self-motion-manifold selector (GC2, GC4, GC6): picks the shoulder/
    // elbow/wrist IK branch for the redundant arm. Fixed for the whole
    // planning problem, so it lives here instead of being threaded through
    // parameterized_ik's input.
    static constexpr std::size_t num_smm_parameters = 3;
    inline static thread_local std::array<float, num_smm_parameters> smm = {1.0, 1.0, -1.0};



    static constexpr std::array<std::string_view, ambient_dimension> joint_names = {"{{join(joint_names, "\", \"")}}"};
    static constexpr char* end_effector = "{{end_effector}}";

    using Configuration = FloatVector<dimension>;
    using ConfigurationArray = std::array<FloatT, dimension>;

    using Sample = FloatVector<sample_dimension>;

    struct alignas(FloatVectorAlignment) ConfigurationBuffer
        : std::array<float, Configuration::num_scalars_rounded>
    {
    };

    template <std::size_t rake>
    using ConfigurationBlock = FloatVector<rake, dimension>;

    template <std::size_t rake>
    using AmbientConfigurationBlock = FloatVector<rake, ambient_dimension>;
    using AmbientConfiguration = FloatVector<ambient_dimension>;
    using AmbientConfigurationArray = std::array<FloatT, ambient_dimension>;

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

    using NNKey = std::tuple<vamp::planning::NNFloatArray<3>, Eigen::Quaternion<float>>;

    using NNSpace = unc::robotics::nigh::metric::CartesianSpace<
    unc::robotics::nigh::metric::Space<vamp::planning::NNFloatArray<3>, unc::robotics::nigh::metric::LP<2>>,
    unc::robotics::nigh::metric::Space<Eigen::Quaternion<float>, unc::robotics::nigh::metric::SO3>>;

    static inline auto nn_key(float *cfg_ptr) noexcept -> NNKey
    {
        return NNKey{
            vamp::planning::NNFloatArray<3>{cfg_ptr},
            Eigen::Quaternion<float>(cfg_ptr[6], cfg_ptr[3], cfg_ptr[4], cfg_ptr[5])};
    }

    static inline auto sample(const Sample &x_in) -> Configuration
    {
        ConfigurationBuffer y;
        std::array<float, {{se3_sampler_code_vars}}> v;
        const auto x = x_in.to_array();
        {{se3_sampler_code}}
        return Configuration(y.data());
    }


    static inline auto distance(const Configuration &a_in, const Configuration &b_in) -> float
    {
        std::array<float, {{distance_code_vars}}> v;
        std::array<float, 1> y;
        const auto a = a_in.to_array();
        const auto b = b_in.to_array();
        {{distance_code}}
        return y[0];
    }

    static inline auto interpolate(const Configuration &a_in, const Configuration &b_in, float t) -> Configuration
    {
        std::array<float, {{interpolate_code_vars}}> v;
        ConfigurationBuffer y;
        const auto a = a_in.to_array();
        const auto b = b_in.to_array();
        {{interpolate_code}}
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
        {% if interpolate_block_code_vars > 0 %}std::array<V, {{interpolate_block_code_vars}}> v;{% endif %}
        {{interpolate_block_code}}
    }

    template <std::size_t rake>
    static inline void sphere_fk(const AmbientConfigurationBlock<rake> &x, Spheres<rake> &out) noexcept
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
            const AmbientConfigurationBlock<rake> &x) noexcept -> Debug
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

        return output;
    }

    template <std::size_t rake>
        static inline bool fkcc(
            const vamp::collision::Environment<FloatVector<rake>> &environment,
            const AmbientConfigurationBlock<rake> &x) noexcept
    {
        std::array<FloatVector<rake, 1>, {{ccfk_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{ccfk_code_output}}> y;

        {{ccfk_code}}
        {% include "ccfk" %}

        return true;
    }

    template <std::size_t rake>
    static inline bool fkcc_attach(
        const vamp::collision::Environment<FloatVector<rake>> &environment,
        const AmbientConfigurationBlock<rake> &x) noexcept
    {
        std::array<FloatVector<rake, 1>, {{ccfkee_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{ccfkee_code_output}}> y;

        {{ccfkee_code}}
        {% include "ccfk" %}

        // attaching at {{ end_effector }}
        set_attachment_pose(environment, to_isometry(&y[{{ccfkee_code_output - 12}}]));

        //
        // attachment vs. environment collisions
        //
        if (attachment_environment_collision(environment))
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
                                                        y[{{(n_spheres + link_bs) * 4 + 3}}]))
        {
            {% for j in range(length(link_spheres)) %}
            {% set sphere_index = at(link_spheres, j) %}
            if (attachment_sphere_collision<decltype(x[0])>(environment,
                                                            y[{{sphere_index * 4 + 0}}],
                                                            y[{{sphere_index * 4 + 1}}],
                                                            y[{{sphere_index * 4 + 2}}],
                                                            y[{{sphere_index * 4 + 3}}]))
            {
                return false;
            }
            {% endfor %}
        }
        {% endfor %}

        return true;
    }

    static inline auto eefk(const std::array<float, {{ambient_nq}}> &x) noexcept -> Eigen::Isometry3f
    {
        std::array<float, {{eefk_code_vars}}> v;
        std::array<float, {{eefk_code_output}}> y;

        {{eefk_code}}

        return to_isometry(y.data());
    }

    template <typename InputVector, std::size_t rake>
    static inline auto parameterized_ik(const InputVector &x) noexcept -> std::pair<bool, AmbientConfigurationBlock<rake>>
    {
        using V = FloatVector<rake, 1>;
        // x is the pose (+psi); the self-motion-manifold selector comes from `smm`.
        const auto &pose = x;
        const auto psi = x[7];

        FloatVector<rake, {{param_ik_code_vars}}> v;
        FloatVector<rake, {{param_ik_code_output}}> y;

        {{param_ik_code}}

        // Check if y are within joint limits
        {% for index in range(n_q - 1) %}
        if ((y[{{index}}] < {{ at(bound_lower, index) }}).any() || (y[{{index}}] > {{ at(bound_lower, index) }} + {{ at(bound_range, index) }}).any())
        {
            return {false, y};
        }
        {% endfor %}

        return {true, y};
    }


};
}

// NOLINTEND(*-magic-numbers)
