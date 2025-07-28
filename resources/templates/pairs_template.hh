// Generated allowed_link_pairs for {{name}}
// Number of pairs: {{length(allowed_link_pairs)}}

{% for i in range(length(allowed_link_pairs)) %}
{% set pair = at(allowed_link_pairs, i) %}
// Pair {{i}}: {{ at(pair, 0) }}, {{ at(pair, 1) }}
{% endfor %}
