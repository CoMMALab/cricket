{% set robot_template_name = capitalize(name) %}
{% set name = lower(name) %}
{% set name_upper = upper(name) %}

struct {{robot_template_name}}{

    static constexpr auto name = "{{name}}";
    static constexpr std::size_t dimension = {{n_q}};
    using Configuration = std::array<float, dimension>;

    __device__ static constexpr float get_s_m(int i) {
        constexpr float values[] = {
            {{join(bound_range, ", ")}}
        };
        return values[i];
    }
    
    __device__ static constexpr float get_s_a(int i) {
        constexpr float values[] = {
            {{join(bound_lower, ", ")}}
        };
        return values[i];
    }
    
    template<size_t I = 0>
    __device__ __forceinline__ static void scale_cfg_impl(float *q)
    {
        if constexpr (I < dimension) {
            q[I] = q[I] * get_s_m(I) + get_s_a(I);
            scale_cfg_impl<I + 1>(q);
        }
    }

    __device__ __forceinline__ static void scale_cfg(float *q)
    {
        scale_cfg_impl(q);
    }

    inline static void print_robot_config(Configuration &cfg) {
        for (int i = 0; i < dimension; i++) {
            std::cout << cfg[i] << ' ';
        }
        std::cout << '\n';
    };
};