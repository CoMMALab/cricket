{% set robot_template_name = capitalize(name) %}
{% set name = lower(name) + "_approx" %}
{% set name_upper = upper(name) %}
{% set n_matrices_saved = n_joints_with_multiple_children + 1 %}

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
#define BATCH_SIZE {{batch_size}}

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

__device__ __constant__ int {{name}}_sphere_to_joint[{{length(sphere_to_joint)}}] = {
    {% for i in range(length(sphere_to_joint)) %}{{ at(sphere_to_joint, i) }}{% if i < length(sphere_to_joint) - 1 %},
    {% endif %}{% endfor %}
};

__device__ __constant__ int {{name}}_flattened_joint_to_spheres[{{length(flattened_joint_to_spheres)}}] = {
    {% for i in range(length(flattened_joint_to_spheres)) %}{{ at(flattened_joint_to_spheres, i) }}{% if i < length(flattened_joint_to_spheres) - 1 %},
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
__device__ void fk_approx<ppln::robots::{{robot_template_name}}>(
    const float* q,
    volatile float* sphere_pos_approx, // {{length(spheres_array)}} spheres x {{batch_size}} robots x 3 coordinates (each column is a robot)
    float *T, // {{batch_size}} robots x {{n_matrices_saved}} x 4x4 transform matrix , column major
    const int tid
)
{
    // every 4 threads are responsible for one column of the transform matrix T
    // make_transform will calculate the necessary column of T_step needed for the thread
    const int col_ind = tid % 4;
    const int batch_ind = tid / 4;

    int T_offset = batch_ind * {{n_matrices_saved + 1}} * 16;
    float T_step_col[4]; // 4x1 column of the joint transform matrix for this thread
    float *T_base = T + T_offset; // 4x4 transform matrix for the batch
    
    #pragma unroll
    for (int i = 0; i < {{n_matrices_saved + 1}}; ++i) {
        float *T_col_i = T + i * 16 + col_ind * 4;
        for (int r=0; r<4; r++) {
            T_col_i[r] = 0.0f;
        }
        T_col_i[col_ind] = 1.0f;
    }

    int joint_to_sphere_ind = 0;

    // loop through each joint, accumulate transformation matrix, and update sphere positions
    for (int j = 0; j < {{name_upper}}_JOINT_COUNT; ++j) {
        int i = {{name}}_dfs_order[j];
        float T_col_tmp[4];
        int parent_idx = {{name}}_joint_parents[i];
        int T_memory_idx_parent = {{name}}_T_memory_idx[parent_idx];
        int T_memory_idx = {{name}}_T_memory_idx[i];
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
                T_col_tmp[r] = dot4_col(&T_base[T_memory_idx_parent*16 + r], T_step_col);
            }
            for (int r=0; r<4; r++){
                T_base[T_memory_idx*16 + col_ind*4 + r] = T_col_tmp[r];
            }
        }
        __syncwarp();
        while ({{name}}_flattened_joint_to_spheres[joint_to_sphere_ind] != -1) {
            int sphere_ind = {{name}}_flattened_joint_to_spheres[joint_to_sphere_ind];
            if (col_ind < 3) {
                // sphere sphere_ind, robot batch_ind (BATCH_SIZE robots), coord col_ind
                sphere_pos_approx[sphere_ind * BATCH_SIZE * 3 + batch_ind * 3 + col_ind] = 
                    T_base[col_ind] * {{name}}_spheres_array[sphere_ind].x +
                    T_base[col_ind + M] * {{name}}_spheres_array[sphere_ind].y +
                    T_base[col_ind + M*2] * {{name}}_spheres_array[sphere_ind].z +
                    T_base[col_ind + M*3];
            }
            joint_to_sphere_ind++;
        }
        joint_to_sphere_ind++;
    }
}

// 4 threads per discretized motion for self-collision check
template <>
__device__ bool self_collision_check_approx<ppln::robots::{{robot_template_name}}>(volatile float* sphere_pos_approx, volatile int* joint_in_collision, const int tid){
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;

    for (int i = thread_ind; i < {{name_upper}}_SELF_CC_RANGE_COUNT; i+=4) {
        int sphere_1_ind = {{name}}_self_cc_ranges[i][0];
        float sphere_1[3] = {
            sphere_pos_approx[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 0],
            sphere_pos_approx[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 1],
            sphere_pos_approx[sphere_1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 2]
        };
        for (int j = {{name}}_self_cc_ranges[i][1]; j <= {{name}}_self_cc_ranges[i][2]; j++) {
            float sphere_2[3] = {
                sphere_pos_approx[j * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                sphere_pos_approx[j * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                sphere_pos_approx[j * BATCH_SIZE * 3 + batch_ind * 3 + 2]
            };
            if (sphere_sphere_self_collision(
                sphere_1[0], sphere_1[1], sphere_1[2], {{name}}_spheres_array[sphere_1_ind].w,
                sphere_2[0], sphere_2[1], sphere_2[2], {{name}}_spheres_array[j].w
            )){
                atomicAdd((int*)&joint_in_collision[20*batch_ind + {{name}}_sphere_to_joint[sphere_1_ind]], 1);
                return false;
            }
        } 
    }
    return true;
}

// 4 threads per discretized motion for env collision check
template <>
__device__ bool env_collision_check_approx<ppln::robots::{{robot_template_name}}>(volatile float* sphere_pos_approx, volatile int* joint_in_collision, ppln::collision::Environment<float> *env, const int tid){
    const int thread_ind = tid % 4;
    const int batch_ind = tid / 4;
    bool out = true;
    
    #pragma unroll
    for (int i={{name_upper}}_SPHERE_COUNT/4*thread_ind; i<{{name_upper}}_SPHERE_COUNT/4*(thread_ind+1); i++){
        // sphere i, robot batch_ind ({{batch_size}} robots)
        if (sphere_environment_in_collision(
            env,
            sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
            sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
            sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
            {{name}}_spheres_array[i].w
        )) {
            atomicAdd((int*)&joint_in_collision[20*batch_ind + {{name}}_sphere_to_joint[i]],1);
            out=false;
        } 
    }

    int i = {{name_upper}}_SPHERE_COUNT-1-thread_ind;
    if (sphere_environment_in_collision(
        env,
        sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
        sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
        sphere_pos_approx[i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
        {{name}}_spheres_array[i].w
    )) {
        atomicAdd((int*)&joint_in_collision[20*batch_ind + {{name}}_sphere_to_joint[i]],1);
        out=false;
    }
    return out;
}