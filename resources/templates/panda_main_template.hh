{% set robot_template_name = capitalize(name) %}
{% set name = lower(name) %}
{% set name_upper = upper(name) %}

#define {{name_upper}}_SPHERE_COUNT {{length(spheres_array)}}
#define {{name_upper}}_JOINT_COUNT {{length(joint_matrices)}}
#define {{name_upper}}_SELF_CC_RANGE_COUNT {{length(self_cc_ranges)}}
#define FIXED -1
#define X_PRISM 0
#define Y_PRISM 1
#define Z_PRISM 2
#define X_ROT 3
#define Y_ROT 4
#define Z_ROT 5

__device__ __constant__ float4 {{name}}_spheres_array[{{length(spheres_array)}}] = {
    {% for i in range(length(spheres_array)) %}{% set sphere = at(spheres_array, i) %}{ {{ round(at(sphere, 0), 6) }}f, {{ round(at(sphere, 1), 6) }}f, {{ round(at(sphere, 2), 6) }}f, {{ round(at(sphere, 3), 6) }}f }{% if i < length(spheres_array) - 1 %},
    {% endif %}{% endfor %}
};

__device__ __constant__ float {{name}}_fixed_transforms[] = {
    {% for i in range(length(joint_matrices)) %}// joint {{i}}
    {% set matrix = at(joint_matrices, i) %}{% for j in range(4) %}{% set row = at(matrix, j) %}{% for k in range(4) %}{{ round(at(row, k), 6) }}{% if k < 3 %}, {% endif %}{% if k == 3 %},
    {% endif %}{% if j == 3 and k == 3 %}
    {% endif %}{% endfor %}{% endfor %}{% endfor %}
};

__device__ __constant__ int {{name}}_sphere_to_joint[] = {
    {% for i in range(length(sphere_to_joint)) %}{{ at(sphere_to_joint, i) }}{% if i < length(sphere_to_joint) - 1 %},
    {% endif %}{% endfor %}
};

__device__ __constant__ int {{name}}_joint_types[] = {
    {% for i in range(length(joint_types)) %}{{ at(joint_types, i) }}{% if i < length(joint_types) - 1 %},
    {% endif %}{% endfor %}
};

__device__ __constant__ int {{name}}_self_cc_ranges[{{length(self_cc_ranges)}}][3] = {
    {% for i in range(length(self_cc_ranges)) %}{% set range = at(self_cc_ranges, i) %}{ {{ at(range, 0) }}, {{ at(range, 1) }}, {{ at(range, 2) }} }{% if i < length(self_cc_ranges) - 1 %},
    {% endif %}{% endfor %}
};

__device__ __constant__ int {{name}}_joint_parents[{{length(joint_parents)}}] = {
    {% for i in range(length(joint_parents)) %}{{ at(joint_parents, i) }}{% if i < length(joint_parents) - 1 %},
    {% endif %}{% endfor %}
};

__device__ __constant__ int {{name}}_T_memory_idx[{{length(T_memory_idx)}}] = {
    {% for i in range(length(T_memory_idx)) %}{{ at(T_memory_idx, i) }}{% if i < length(T_memory_idx) - 1 %},
    {% endif %}{% endfor %}
};

__device__ __constant__ int {{name}}_dfs_order[{{length(dfs_order)}}] = {
    {% for i in range(length(dfs_order)) %}{{ at(dfs_order, i) }}{% if i < length(dfs_order) - 1 %},
    {% endif %}{% endfor %}
};


template <>
__device__ void fk<ppln::robots::{{robot_template_name}}>(
    const float* q,
    volatile float* sphere_pos, // {{length(spheres_array)}} spheres x 16 robots x 3 coordinates
    float *T, // 16 robots x 4x4 transform matrix
    const int tid
)
{
    // every 4 threads are responsible for one column of the transform matrix T
    // make_transform will calculate the necessary column of T_step needed for the thread
    const int col_ind = tid % 4;
    const int batch_ind = tid / 4;
    int transformed_sphere_ind = 0;

    int T_offset = batch_ind * 16;
    float T_step_col[4]; // 4x1 column of the joint transform matrix for this thread
    float *T_base = T + T_offset; // 4x4 transform matrix for the batch
    float *T_col = T_base + col_ind*4; // 1x4 column (column major) of the transform matrix for this thread

    for (int r=0; r<4; r++){
        T_col[r] = 0;
    }
    T_col[col_ind] = 1;

    // loop through each joint, accumulate transformation matrix, and update sphere positions
    for (int i = 0; i < {{name_upper}}_JOINT_COUNT; ++i) {
        if (i > 0) {
            int ft_addr_start = i * 16;
            int joint_type = {{name}}_joint_types[i];
            if (joint_type <= Z_PRISM) {
                prism_fn(&{{name}}_fixed_transforms[ft_addr_start], q[i - 1], col_ind, T_step_col, joint_type);
            }
            else if (joint_type == X_ROT) {
                xrot_fn(&{{name}}_fixed_transforms[ft_addr_start], q[i - 1], col_ind, T_step_col);
            }
            else if (joint_type == Y_ROT) { 
                yrot_fn(&{{name}}_fixed_transforms[ft_addr_start], q[i - 1], col_ind, T_step_col);
            }
            else if (joint_type == Z_ROT) {
                zrot_fn(&{{name}}_fixed_transforms[ft_addr_start], q[i - 1], col_ind, T_step_col);
            }

            for (int r=0; r<4; r++){
                T_col[r] = dot4_col(&T_base[r], T_step_col);
            }
        }

        while ({{name}}_sphere_to_joint[transformed_sphere_ind]==i) {
            if (col_ind < 3) {
                // sphere transformed_sphere_ind, robot batch_ind (16 robots), coord col_ind
                sphere_pos[transformed_sphere_ind * 16 * 3 + batch_ind * 3 + col_ind] = 
                    T_base[col_ind] * {{name}}_spheres_array[transformed_sphere_ind].x +
                    T_base[col_ind + M] * {{name}}_spheres_array[transformed_sphere_ind].y +
                    T_base[col_ind + M*2] * {{name}}_spheres_array[transformed_sphere_ind].z +
                    T_base[col_ind + M*3];
            }
            transformed_sphere_ind++;
        }
    }
}

// 4 threads per discretized motion for self-collision check
template <>
__device__ bool self_collision_check<ppln::robots::{{robot_template_name}}>(volatile float* sphere_pos, volatile int* joint_in_collision, const int tid){
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;

    for (int i = thread_ind; i < {{name_upper}}_SELF_CC_RANGE_COUNT; i += 4) {
        int sphere_1_ind = {{name}}_self_cc_ranges[i][0];
        if (joint_in_collision[20*batch_ind + {{name}}_sphere_to_joint[sphere_1_ind]] == 0) continue;
        float sphere_1[3] = {
            sphere_pos[sphere_1_ind * 16 * 3 + batch_ind * 3 + 0],
            sphere_pos[sphere_1_ind * 16 * 3 + batch_ind * 3 + 1],
            sphere_pos[sphere_1_ind * 16 * 3 + batch_ind * 3 + 2]
        };
        for (int j = {{name}}_self_cc_ranges[i][1]; j <= {{name}}_self_cc_ranges[i][2]; j++) {
            float sphere_2[3] = {
                sphere_pos[j * 16 * 3 + batch_ind * 3 + 0],
                sphere_pos[j * 16 * 3 + batch_ind * 3 + 1],
                sphere_pos[j * 16 * 3 + batch_ind * 3 + 2]
            };
            if (sphere_sphere_self_collision(
                sphere_1[0], sphere_1[1], sphere_1[2], {{name}}_spheres_array[sphere_1_ind].w,
                sphere_2[0], sphere_2[1], sphere_2[2], {{name}}_spheres_array[j].w
            )){
                return false;
            }
        }
    }
    return true;

}

// 4 threads per discretized motion for env collision check
template <>
__device__ bool env_collision_check<ppln::robots::{{robot_template_name}}>(volatile float* sphere_pos, volatile int* joint_in_collision, ppln::collision::Environment<float> *env, const int tid){
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;

    for (int i = thread_ind; i < {{name_upper}}_SPHERE_COUNT; i += 4){
        // sphere i, robot batch_ind (16 robots)
        if (joint_in_collision[20*batch_ind + {{name}}_sphere_to_joint[i]] > 0 && 
            sphere_environment_in_collision(
                env,
                sphere_pos[i * 16 * 3 + batch_ind * 3 + 0],
                sphere_pos[i * 16 * 3 + batch_ind * 3 + 1],
                sphere_pos[i * 16 * 3 + batch_ind * 3 + 2],
                {{name}}_spheres_array[i].w
            )
        ) {
            return false;
        } 
    }
    return true;
}