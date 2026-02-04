{% for i in range(length(links_with_geometry)) %}
{% set array_index = length(links_with_geometry) - i - 1 %}
{% set link_index = at(links_with_geometry, array_index) %}
{% set link_spheres = at(per_link_spheres, link_index) %}
{% set bs_loc = (n_spheres + array_index) * 4 %}

//
// environment vs. robot collisions
//

// {{ at(link_names, link_index) }}
if sphere_environment_in_collision(environment,
                                    y[{{bs_loc + 0}}],
                                    y[{{bs_loc + 1}}],
                                    y[{{bs_loc + 2}}],
                                    y[{{bs_loc + 3}}])
{
    {% for j in range(length(link_spheres)) %}
    {% set sphere_loc = at(link_spheres, j) * 4 %}
    if sphere_environment_in_collision(environment,
                                        y[{{ sphere_loc + 0 }}],
                                        y[{{ sphere_loc + 1 }}],
                                        y[{{ sphere_loc + 2 }}],
                                        y[{{ sphere_loc + 3 }}])
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
if sphere_sphere_self_collision(
    y[{{link_1_bs_loc + 0}}],
    y[{{link_1_bs_loc + 1}}],
    y[{{link_1_bs_loc + 2}}],
    y[{{link_1_bs_loc + 3}}],
    y[{{link_2_bs_loc + 0}}],
    y[{{link_2_bs_loc + 1}}],
    y[{{link_2_bs_loc + 2}}],
    y[{{link_2_bs_loc + 3}}]
) {
    {% for j in range(length(link_1_spheres)) %}
    {% for k in range(length(link_2_spheres)) %}

    {% set sphere_1_loc = at(link_1_spheres, j) %}
    {% set sphere_2_loc = at(link_2_spheres, k) %}

    if sphere_sphere_self_collision(
        y[{{ sphere_1_loc * 4 + 0}} ],
        y[{{ sphere_1_loc * 4 + 1}} ],
        y[{{ sphere_1_loc * 4 + 2}} ],
        y[{{ sphere_1_loc * 4 + 3}} ],
        y[{{ sphere_2_loc * 4 + 0}} ],
        y[{{ sphere_2_loc * 4 + 1}} ],
        y[{{ sphere_2_loc * 4 + 2}} ],
        y[{{ sphere_2_loc * 4 + 3}} ]
    ) {
        return false;
    }

    {% endfor %}
    {% endfor %}
}
{% endfor %}
