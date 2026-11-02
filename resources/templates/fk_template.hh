#pragma once

#include <vamp/vector.hh>
#include <vamp/vector/math.hh>
#include <vamp/collision/environment.hh>
#include <vamp/collision/validity.hh>

// NOLINTBEGIN(*-magic-numbers)
namespace vamp::robots
{
struct {{name}}
{
    static constexpr const char *name = "{{lower(name)}}";
    static constexpr std::size_t dimension = {{n_q}};
    static constexpr std::size_t sample_dimension = {{n_u}};
    static constexpr std::size_t n_spheres = {{n_spheres}};
    static constexpr std::size_t n_cuboids = {{n_cuboids}};
    static constexpr float min_radius = {{min_radius}};
    static constexpr float max_radius = {{max_radius}};
    static constexpr float min_radius_mobile = {{min_radius_mobile}};
    static constexpr float max_radius_mobile = {{max_radius_mobile}};
    static constexpr float min_bounding_radius_mobile = {{min_bounding_radius_mobile}};
    static constexpr float max_bounding_radius_mobile = {{max_bounding_radius_mobile}};
    static constexpr std::array<float, 3> base_position = { {{ join(base_position, ", ") }} };
    static constexpr std::size_t resolution = {{resolution}};
    static constexpr bool euclidean = {{euclidean}};

    static constexpr std::array<std::string_view, dimension> joint_names = {"{{join(joint_names, "\", \"")}}"};
    static constexpr const char *end_effector = "{{end_effector}}";

    using Configuration = FloatVector<dimension>;
    using ConfigurationArray = std::array<FloatT, dimension>;
    using Sample = FloatVector<sample_dimension>;
    
    static constexpr std::array<std::size_t, {{length(so3_offsets)}}> so3_offsets = { {{join(so3_offsets, ", ")}} };

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

    {% if n_cuboids > 0 %}
    // Cuboid collision geometry, one row per cuboid. The layout matches vamp::collision::OrientedBox:
    // center, three unit axes, and half-lengths along each axis.
    template <std::size_t rake>
    struct Cuboids
    {
        FloatVector<rake, n_cuboids> x;
        FloatVector<rake, n_cuboids> y;
        FloatVector<rake, n_cuboids> z;
        FloatVector<rake, n_cuboids> axis_1_x;
        FloatVector<rake, n_cuboids> axis_1_y;
        FloatVector<rake, n_cuboids> axis_1_z;
        FloatVector<rake, n_cuboids> axis_2_x;
        FloatVector<rake, n_cuboids> axis_2_y;
        FloatVector<rake, n_cuboids> axis_2_z;
        FloatVector<rake, n_cuboids> axis_3_x;
        FloatVector<rake, n_cuboids> axis_3_y;
        FloatVector<rake, n_cuboids> axis_3_z;
        FloatVector<rake, n_cuboids> axis_1_r;
        FloatVector<rake, n_cuboids> axis_2_r;
        FloatVector<rake, n_cuboids> axis_3_r;
    };
    {% endif %}

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
        ConfigurationBuffer y;
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
        ConfigurationBuffer y;
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

    {% if n_cuboids > 0 %}
    template <std::size_t rake>
    static inline void cuboid_fk(const ConfigurationBlock<rake> &x, Cuboids<rake> &out) noexcept
    {
        std::array<FloatVector<rake, 1>, {{cuboidfk_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{cuboidfk_code_output}}> y;

        {{cuboidfk_code}}

        for (auto i = 0U; i < {{n_cuboids}}; ++i)
        {
            out.x[i] = y[i * 15 + 0];
            out.y[i] = y[i * 15 + 1];
            out.z[i] = y[i * 15 + 2];
            out.axis_1_x[i] = y[i * 15 + 3];
            out.axis_1_y[i] = y[i * 15 + 4];
            out.axis_1_z[i] = y[i * 15 + 5];
            out.axis_2_x[i] = y[i * 15 + 6];
            out.axis_2_y[i] = y[i * 15 + 7];
            out.axis_2_z[i] = y[i * 15 + 8];
            out.axis_3_x[i] = y[i * 15 + 9];
            out.axis_3_y[i] = y[i * 15 + 10];
            out.axis_3_z[i] = y[i * 15 + 11];
            out.axis_1_r[i] = y[i * 15 + 12];
            out.axis_2_r[i] = y[i * 15 + 13];
            out.axis_3_r[i] = y[i * 15 + 14];
        }
    }
    {% endif %}

    // For fkcc_debug: the first element has one entry per sphere, followed by one per cuboid. The second
    // element lists colliding pairs of robot geometry; indices < n_spheres are spheres and
    // n_spheres + i is cuboid i.
    using Debug = std::pair<std::vector<std::vector<std::string>>, std::vector<std::pair<std::size_t, std::size_t>>>;

    template <std::size_t rake>
        static inline auto fkcc_debug(
            const vamp::collision::Environment<FloatVector<rake>> &environment,
            const ConfigurationBlock<rake> &x) noexcept -> Debug
    {
        std::array<FloatVector<rake, 1>, {{ccfk_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{ccfk_code_output}}> y;

        {{ccfk_code}}

        {% if n_cuboids > 0 %}
        {% for c in range(n_cuboids) %}
        {% set cuboid_loc = cuboid_buffer_offset + c * 15 %}
        const vamp::collision::OrientedBox<FloatVector<rake>> cuboid_{{ c }}(
            {% for k in range(15) %}y[{{ cuboid_loc + k }}]{% if k < 14 %}, {% endif %}{% endfor %});
        {% endfor %}
        {% endif %}

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

        {% for c in range(n_cuboids) %}
        output.first.emplace_back(vamp::cuboid_environment_get_collisions(environment, cuboid_{{ c }}));
        {% endfor %}

        {% for i in range(length(allowed_link_pairs)) %}
        {% set pair = at(allowed_link_pairs, i) %}
        {% set link_1_index = at(pair, 0) %}
        {% set link_2_index = at(pair, 1) %}
        {% set link_1_spheres = at(per_link_spheres, link_1_index) %}
        {% set link_2_spheres = at(per_link_spheres, link_2_index) %}
        {% set link_1_cuboids = at(per_link_cuboids, link_1_index) %}
        {% set link_2_cuboids = at(per_link_cuboids, link_2_index) %}

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

        {% for j in range(length(link_1_spheres)) %}
        {% for k in range(length(link_2_cuboids)) %}
        {% set sphere_1_loc = at(link_1_spheres, j) %}
        {% set cuboid_2 = at(link_2_cuboids, k) %}
        if (vamp::sphere_cuboid_self_collision(cuboid_{{ cuboid_2 }},
                                               y[{{ sphere_1_loc * 4 + 0}} ],
                                               y[{{ sphere_1_loc * 4 + 1}} ],
                                               y[{{ sphere_1_loc * 4 + 2}} ],
                                               y[{{ sphere_1_loc * 4 + 3}} ]))
        {
            output.second.emplace_back({{ sphere_1_loc }}, n_spheres + {{ cuboid_2 }});
        }
        {% endfor %}
        {% endfor %}

        {% for j in range(length(link_1_cuboids)) %}
        {% for k in range(length(link_2_spheres)) %}
        {% set cuboid_1 = at(link_1_cuboids, j) %}
        {% set sphere_2_loc = at(link_2_spheres, k) %}
        if (vamp::sphere_cuboid_self_collision(cuboid_{{ cuboid_1 }},
                                               y[{{ sphere_2_loc * 4 + 0}} ],
                                               y[{{ sphere_2_loc * 4 + 1}} ],
                                               y[{{ sphere_2_loc * 4 + 2}} ],
                                               y[{{ sphere_2_loc * 4 + 3}} ]))
        {
            output.second.emplace_back(n_spheres + {{ cuboid_1 }}, {{ sphere_2_loc }});
        }
        {% endfor %}
        {% endfor %}

        {% for j in range(length(link_1_cuboids)) %}
        {% for k in range(length(link_2_cuboids)) %}
        {% set cuboid_1 = at(link_1_cuboids, j) %}
        {% set cuboid_2 = at(link_2_cuboids, k) %}
        if (vamp::cuboid_cuboid_self_collision(cuboid_{{ cuboid_1 }}, cuboid_{{ cuboid_2 }}))
        {
            output.second.emplace_back(n_spheres + {{ cuboid_1 }}, n_spheres + {{ cuboid_2 }});
        }
        {% endfor %}
        {% endfor %}
        {% endfor %}

        return output;
    }

    template <std::size_t rake>
        static inline bool fkcc(
            const vamp::collision::Environment<FloatVector<rake>> &environment,
            const ConfigurationBlock<rake> &x) noexcept
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
        const ConfigurationBlock<rake> &x) noexcept
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
        {% set link_cuboids = at(per_link_cuboids, link_index) %}

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
            {% for j in range(length(link_cuboids)) %}
            if (vamp::attachment_cuboid_collision(environment, cuboid_{{ at(link_cuboids, j) }})) [[unlikely]]
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
};
}

// NOLINTEND(*-magic-numbers)
