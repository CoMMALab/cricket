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
    static constexpr char* name = "{{lower(name)}}";
    static constexpr std::size_t dimension = {{n_q}};
    static constexpr std::size_t n_spheres = {{n_spheres}};
    static constexpr float min_radius = {{min_radius}};
    static constexpr float max_radius = {{max_radius}};
    static constexpr std::size_t resolution = {{resolution}};
    static constexpr std::size_t n_eef = {{num_end_effectors}};
    static constexpr std::size_t num_bounding_spheres = {{num_bounding_spheres}};

    static constexpr std::array<std::string_view, dimension> joint_names = {"{{join(joint_names, "\", \"")}}"};
    static constexpr std::array<std::string_view, {{num_end_effectors}}> end_effectors ={"{{join(end_effectors, "\", \"")}}"};

    using Configuration = FloatVector<dimension>;
    using ConfigurationArray = std::array<FloatT, dimension>;

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

        // attaching at {{ end_effectors }}
        // provide the eef pose of all end effectors.
        set_attachment_pose(environment, to_isometries<{{num_end_effectors}}>(&y[{{ccfkee_code_output - 12 * num_end_effectors}}]));

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

    static inline auto eefk(const std::array<float, {{n_q}}> &x) noexcept -> std::array<Eigen::Isometry3f, {{num_end_effectors}}>
    {
        std::array<float, {{eefk_code_vars}}> v;
        std::array<float, {{eefk_code_output}}> y;

        {{eefk_code}}

        return to_isometries<{{num_end_effectors}}>(y.data());
    }

    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto tsr_error(const InputVector &x, OutputVector &out)
    {
        std::array<FloatVector<rake, 1>, {{tsr_error_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{tsr_error_code_output}}> y;

        {{tsr_error_code}}

        for(size_t i = 0; i < {{tsr_error_code_output}}; i++)
            out[i] = y[i];
    }

    template <std::size_t rake, typename InputVector>
    static inline auto solve_tsr_error_lm_inner(const InputVector &x, ConfigurationBlock<rake> &y)
    {
        std::array<FloatVector<rake, 1>, {{solve_tsr_error_lm_inner_code_vars}}> v;

        {{solve_tsr_error_lm_inner_code}}

    }

    template <std::size_t rake, typename InputVector>
    static inline auto solve_tsr_error_lm_outer(const InputVector &x, ConfigurationBlock<rake> &y)
    {
        std::array<FloatVector<rake, 1>, {{solve_tsr_error_lm_outer_code_vars}}> v;

        {{solve_tsr_error_lm_outer_code}}
    }

    template <std::size_t rake, typename InputVector>
    static inline auto solve_tsr_error_gradient_descent(const InputVector &x, ConfigurationBlock<rake> &y)
    {
        std::array<FloatVector<rake, 1>, {{solve_tsr_error_gradient_descent_code_vars}}> v;

        {{solve_tsr_error_gradient_descent_code}}
    }


    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto compute_com(const InputVector &x, OutputVector &out)
    {
        std::array<FloatVector<rake, 1>, {{CoM_code_vars}}> v;
        std::array<FloatVector<rake, 1>, {{CoM_code_output}}> y;

        {{CoM_code}}

        for(size_t i = 0; i < {{CoM_code_output}}; i++)
            out[i] = y[i];
    }

    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto com_constraint_error(const InputVector &com_jac_polygons, size_t n, OutputVector &out)
    {
        // input com_jac_polygons is a 3 + (3 * robot.nq) + (n * 2) where n is the number of polygons

        const size_t polygon_offset = 3 + 3 * dimension;
        // initialize x, com_constraint_grad and out
        FloatVector<rake, 4 + 3> x;
        x[0] = com_jac_polygons[0];
        x[1] = com_jac_polygons[1];
        x[2] = com_jac_polygons[2];

        FloatVector<rake, 2 * 3> com_constraint_grad;
        com_constraint_grad[0] = 0.0;
        com_constraint_grad[1] = 0.0;
        com_constraint_grad[2] = 0.0;
        com_constraint_grad[3] = 0.0;
        com_constraint_grad[4] = 0.0;
        com_constraint_grad[5] = 0.0;


        // initialize the errors to be zero
        out[2 * dimension + 0] = 0.0;
        out[2 * dimension + 1] = 0.0;

        for(size_t polygon_idx = 0; polygon_idx < n; polygon_idx++){

            const auto polygon_next_idx = (polygon_idx + 1) % 4;
            x[3 + 0] = com_jac_polygons[polygon_offset + 2 * polygon_idx + 0];
            x[3 + 1] = com_jac_polygons[polygon_offset + 2 * polygon_idx + 1];
            x[3 + 2] = com_jac_polygons[polygon_offset + 2 * polygon_next_idx + 0];
            x[3 + 3] = com_jac_polygons[polygon_offset + 2 * polygon_next_idx + 1];


            std::array<FloatVector<rake, 1>, {{CoM_constraint_code_vars}}> v;
            std::array<FloatVector<rake, 1>, {{CoM_constraint_code_output}}> y;
            {{CoM_constraint_code}}

            // initialize_error
            out[2 * dimension + 0] = out[2 * dimension + 0] + y[0];
            out[2 * dimension + 1] = out[2 * dimension + 1] + y[1];

            for(size_t j = 0; j < 2 * 3; j++)
                com_constraint_grad[j] = com_constraint_grad[j] + y[2 + j];
        }

        // do the actual matrix multiplication
        for(size_t dim_idx = 0; dim_idx < dimension; dim_idx++){
            out[dim_idx] = com_constraint_grad[0] * com_jac_polygons[3 + dim_idx] + com_constraint_grad[1] * com_jac_polygons[3 + dimension + dim_idx] + com_constraint_grad[2] * com_jac_polygons[3 + 2 * dimension + dim_idx];
            out[dimension + dim_idx] = com_constraint_grad[3] * com_jac_polygons[3 + dim_idx] + com_constraint_grad[4] * com_jac_polygons[3 + dimension + dim_idx] + com_constraint_grad[5] * com_jac_polygons[3 + 2 * dimension + dim_idx];
        }
    }


    template <std::size_t rake, typename InputVector>
    static inline auto solve_com_function_lm_inner(const InputVector &x, ConfigurationBlock<rake> &y)
    {
        std::array<FloatVector<rake, 1>, {{solve_com_function_lm_inner_code_vars}}> v;

        {{solve_com_function_lm_inner_code}}

    }

    template <std::size_t rake, typename InputVector>
    static inline auto solve_com_function_lm_outer(const InputVector &x, ConfigurationBlock<rake> &y)
    {
        std::array<FloatVector<rake, 1>, {{solve_com_function_lm_outer_code_vars}}> v;

        {{solve_com_function_lm_outer_code}}
    }

    template <std::size_t rake, typename InputVector>
    static inline auto solve_com_function_gradient_descent(const InputVector &x, ConfigurationBlock<rake> &y)
    {
        std::array<FloatVector<rake, 1>, {{solve_com_function_gradient_descent_code_vars}}> v;

        {{solve_com_function_gradient_descent_code}}
    }



    {% if num_end_effectors > 1 %}
    template <std::size_t rake, typename InputVector, typename OutputVector>
    static inline auto tsr_bimanual_error(const InputVector &x, OutputVector &out)
    {
        std::array<FloatVector<rake, 1>, {{tsr_bimanual_error_code_vars}}> v;
        FloatVector<rake, {{tsr_bimanual_error_code_output}}> y;

        {{tsr_bimanual_error_code}}

        for(size_t i = 0; i < {{tsr_bimanual_error_code_output}}; i++)
            out[i] = y[i];
    }

    template <std::size_t rake, typename InputVector>
    static inline auto solve_tsr_relative_error_lm_inner(const InputVector &x, ConfigurationBlock<rake> &y)
    {
        std::array<FloatVector<rake, 1>, {{solve_relative_tsr_error_lm_inner_code_vars}}> v;

        {{solve_relative_tsr_error_lm_inner_code}}
    }

    template <std::size_t rake, typename InputVector>
    static inline auto solve_tsr_relative_error_lm_outer(const InputVector &x, ConfigurationBlock<rake> &y)
    {
        std::array<FloatVector<rake, 1>, {{solve_relative_tsr_error_lm_outer_code_vars}}> v;

        {{solve_relative_tsr_error_lm_outer_code}}
    }

    template <std::size_t rake, typename InputVector>
    static inline auto solve_tsr_relative_error_gradient_descent(const InputVector &x, ConfigurationBlock<rake> &y)
    {
        std::array<FloatVector<rake, 1>, {{solve_relative_tsr_error_gradient_descent_code_vars}}> v;

        {{solve_relative_tsr_error_gradient_descent_code}}

    }

    template <std::size_t rake, typename InputVector>
    static inline auto solve_2_eef_tsr_error_lm_inner(const InputVector &x, ConfigurationBlock<rake> &y)
    {
        std::array<FloatVector<rake, 1>, {{solve_2_eef_tsr_error_lm_inner_code_vars}}> v;

        {{solve_2_eef_tsr_error_lm_inner_code}}

    }

    template <std::size_t rake, typename InputVector>
    static inline auto solve_2_eef_tsr_error_lm_outer(const InputVector &x, ConfigurationBlock<rake> &y)
    {
        std::array<FloatVector<rake, 1>, {{solve_2_eef_tsr_error_lm_outer_code_vars}}> v;

        {{solve_2_eef_tsr_error_lm_outer_code}}

    }

    template <std::size_t rake, typename InputVector>
    static inline auto solve_2_eef_tsr_error_gradient_descent(const InputVector &x, ConfigurationBlock<rake> &y)
    {
        std::array<FloatVector<rake, 1>, {{solve_2_eef_tsr_error_gradient_descent_code_vars}}> v;

        {{solve_2_eef_tsr_error_gradient_descent_code}}

    }


    {% endif %}
    {% if name == "Digit" %}

        template <std::size_t rake, typename InputVector, typename OutputVector>
        static inline auto closed_link_joints_error(const InputVector &x, OutputVector &out)
        {
            std::array<FloatVector<rake, 1>, {{digit_closed_link_error_code_vars}}> v;
            FloatVector<rake, {{digit_closed_link_error_code_output}}> y;

            {{digit_closed_link_error_code}}

            for(size_t i = 0; i < {{digit_closed_link_error_code_output}}; i++)
                out[i] = y[i];
        }
        // now add all solvers to it
        template <std::size_t rake, typename InputVector>
        static inline auto solve_closed_link_error_lm_inner(const InputVector &x, ConfigurationBlock<rake> &y)
        {
            std::array<FloatVector<rake, 1>, {{solve_digit_closed_link_error_lm_inner_code_vars}}> v;
            {{solve_digit_closed_link_error_lm_inner_code}}
        }

        template <std::size_t rake, typename InputVector>
        static inline auto solve_closed_link_error_lm_outer(const InputVector &x, ConfigurationBlock<rake> &y)
        {
            std::array<FloatVector<rake, 1>, {{solve_digit_closed_link_error_lm_outer_code_vars}}> v;
            {{solve_digit_closed_link_error_lm_outer_code}}
        }

        template <std::size_t rake, typename InputVector>
        static inline auto solve_closed_link_error_gradient_descent(const InputVector &x, ConfigurationBlock<rake> &y)
        {
            std::array<FloatVector<rake, 1>, {{solve_digit_closed_link_error_gradient_descent_code_vars}}> v;
            {{solve_digit_closed_link_error_gradient_descent_code}}
        }

    {% endif %}


};
}

// NOLINTEND(*-magic-numbers)
