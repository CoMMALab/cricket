    //
    // FLASK flat-system (z-space) sibling: z = (q, qdot), planned kinodynamically
    // with LQMT cubics. Rendered into the parent robot struct; the parent is the
    // ambient position-space robot whose kernels this shares.
    //

    struct Flask
    {
        static constexpr const char *name = "flask";
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

        // Ambient position-space sibling (same kinematic structure, dimension =
        // flat_dimension) whose constraint kernels define manifolds for chart-based
        // constrained planning.
        using Ambient = {{name}};

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
        using Spheres = Ambient::Spheres<rake>;

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

        // Optimal duration T* of the LQMT cubic; generic solver in vamp/planning/flask.hh
        static inline auto optimal_time(const Configuration &a_in, const Configuration &b_in) noexcept -> float
        {
            return vamp::planning::flask::optimal_time<Flask>(a_in, b_in);
        }

        // LQMT edge cost C_loc(a -> b) = rho T* + integral |u|^2; asymmetric in (a, b)
        static inline auto cost(const Configuration &a_in, const Configuration &b_in) noexcept -> float
        {
            return vamp::planning::flask::cost<Flask>(a_in, b_in);
        }

        static inline auto cost_grad(const Configuration &a_in, const Configuration &b_in) noexcept
            -> vamp::planning::flask::LQMTCostGrad<dimension>
        {
            return vamp::planning::flask::cost_grad<Flask>(a_in, b_in);
        }

        // Cubic state (y, yd, ydd) at fraction t of duration T; layout [y; yd; ydd]
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

        // Joint torques for a flat state row-stack x = [q; qd; qdd]
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

        // Position/velocity bounds on rows 0..2n-1 and torque limits via RNEA on all 3n rows.
        // Cubic edges can overshoot bounds mid-path even when both endpoints are valid.
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

        // The FK/CC kernels live on the ambient robot: a z-block stacks (y, yd, ydd), so
        // positions are rows 0..flat_dimension-1 and the ambient kernels read them in place.
        template <std::size_t rake>
        static inline void sphere_fk(const ConfigurationBlock<rake> &x, Spheres<rake> &out) noexcept
        {
            Ambient::sphere_fk<rake>(x, out);
        }

        using Debug = Ambient::Debug;

        template <std::size_t rake>
            static inline auto fkcc_debug(
                const vamp::collision::Environment<FloatVector<rake>> &environment,
                const ConfigurationBlock<rake> &x) noexcept -> Debug
        {
            return Ambient::fkcc_debug<rake>(environment, x);
        }

        template <std::size_t rake>
            static inline bool fkcc(
                const vamp::collision::Environment<FloatVector<rake>> &environment,
                const ConfigurationBlock<rake> &x) noexcept
        {
            return limits_check<rake>(x) and Ambient::fkcc<rake>(environment, x);
        }

        template <std::size_t rake>
        static inline bool fkcc_attach(
            const vamp::collision::Environment<FloatVector<rake>> &environment,
            const ConfigurationBlock<rake> &x) noexcept
        {
            return limits_check<rake>(x) and Ambient::fkcc_attach<rake>(environment, x);
        }

        static inline auto eefk(const std::array<float, dimension> &x) noexcept -> Eigen::Isometry3f
        {
            std::array<float, flat_dimension> q;
            std::copy_n(x.begin(), flat_dimension, q.begin());
            return Ambient::eefk(q);
        }
    };
