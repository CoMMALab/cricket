// clang-format off
// {% if compact_collisions %}
//
// environment vs. robot collisions
//
for (const auto &el : cc_env_links)
{
    const auto bs = (n_spheres + el.bs_array_idx) * 4;
    if (sphere_environment_in_collision(environment,
                                        y[bs + 0], y[bs + 1], y[bs + 2], y[bs + 3])) [[unlikely]]
    {
        for (unsigned int k = 0; k < el.body_count; ++k)
        {
            const auto s = cc_env_body_idx[el.body_start + k] * 4;
            if (sphere_environment_in_collision(environment,
                                                y[s + 0], y[s + 1], y[s + 2], y[s + 3])) [[unlikely]]
            {
                return false;
            }
        }
    }
}

//
// robot self-collisions
//
for (const auto &sp : cc_self_pairs)
{
    const auto bs1 = (n_spheres + sp.bs1_idx) * 4;
    const auto bs2 = (n_spheres + sp.bs2_idx) * 4;
    if (sphere_sphere_self_collision<decltype(x[0])>(
            y[bs1 + 0], y[bs1 + 1], y[bs1 + 2], y[bs1 + 3],
            y[bs2 + 0], y[bs2 + 1], y[bs2 + 2], y[bs2 + 3])) [[unlikely]]
    {
        for (unsigned int k = 0; k < sp.pair_count; ++k)
        {
            const auto a = cc_self_pair_a[sp.pair_start + k] * 4;
            const auto b = cc_self_pair_b[sp.pair_start + k] * 4;
            if (sphere_sphere_self_collision<decltype(x[0])>(
                    y[a + 0], y[a + 1], y[a + 2], y[a + 3],
                    y[b + 0], y[b + 1], y[b + 2], y[b + 3])) [[unlikely]]
            {
                return false;
            }
        }
    }
}
{% else %}
{% for i in range(length(environment_links)) %}
{% set array_index = at(environment_links, i) %}
{% set link_index = at(links_with_geometry, array_index) %}
{% set link_spheres = at(per_link_spheres, link_index) %}
{% set bs_loc = (n_spheres + array_index) * 4 %}

//
// environment vs. robot collisions
//

// {{ at(link_names, link_index) }}
if (sphere_environment_in_collision(environment,
                                    y[{{bs_loc + 0}}],
                                    y[{{bs_loc + 1}}],
                                    y[{{bs_loc + 2}}],
                                    y[{{bs_loc + 3}}])) [[unlikely]]
{
    {% for j in range(length(link_spheres)) %}
    {% set sphere_loc = at(link_spheres, j) * 4 %}
    if (sphere_environment_in_collision(environment,
                                        y[{{ sphere_loc + 0 }}],
                                        y[{{ sphere_loc + 1 }}],
                                        y[{{ sphere_loc + 2 }}],
                                        y[{{ sphere_loc + 3 }}])) [[unlikely]]
    {
        return false;
    }
    {% endfor %}
}

{% endfor %}

//
// robot self-collisions
//

{% for i in range(length(allowed_link_pairs)) %}
{% set pair = at(allowed_link_pairs, i) %}
{% set link_1_index = at(pair, 0) %}
{% set link_2_index = at(pair, 1) %}
{% set link_1_bs = at(bounding_sphere_index, link_1_index) %}
{% set link_2_bs = at(bounding_sphere_index, link_2_index) %}
{% set link_1_spheres = at(per_link_spheres, link_1_index) %}
{% set link_2_spheres = at(per_link_spheres, link_2_index) %}
{% set link_1_bs_loc = (n_spheres + link_1_bs) * 4 %}
{% set link_2_bs_loc = (n_spheres + link_2_bs) * 4 %}

// {{ at(link_names, link_1_index) }} vs. {{ at(link_names, link_2_index) }}
if (sphere_sphere_self_collision<decltype(x[0])>(y[{{link_1_bs_loc + 0}}],
                                                 y[{{link_1_bs_loc + 1}}],
                                                 y[{{link_1_bs_loc + 2}}],
                                                 y[{{link_1_bs_loc + 3}}],
                                                 y[{{link_2_bs_loc + 0}}],
                                                 y[{{link_2_bs_loc + 1}}],
                                                 y[{{link_2_bs_loc + 2}}],
                                                 y[{{link_2_bs_loc + 3}}])) [[unlikely]]
{
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
                                                     y[{{ sphere_2_loc * 4 + 3}} ])) [[unlikely]]
    {
        return false;
    }

    {% endfor %}
    {% endfor %}
}
{% endfor %}
{% endif %}