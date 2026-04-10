#!/usr/bin/env python3
import re
import sys

mask_number = 0


def convert_conditionals_to_blend(code):
    """
    Convert C-style conditionals to blend operations using regex transformations.
    
    Transforms patterns:
    if( condition >= 0. ) { target = expr1; } else { target = expr2; }
    
    To:
    auto mask = condition >= 0.;
    target = target.blend(expr1, mask).blend(expr2, ~mask);
    """

    # Pattern 1: Simple conditional assignment with >= 0 comparison
    pattern1 = r'if\(\s*([^)]+)\s*>=\s*0\.\s*\)\s*{\s*([^=]+)=\s*([^;]+);\s*}\s*else\s*{\s*([^=]+)=\s*([^;]+);\s*}'

    def replace_geq_zero(match):
        condition = match.group(1).strip()
        target1 = match.group(2).strip()
        expr1 = match.group(3).strip()
        target2 = match.group(4).strip()
        expr2 = match.group(5).strip()

        # Assume targets are the same variable
        var_name = target1

        global mask_number
        mask_name = f"mask_{mask_number}"
        mask_number += 1

        return f"""auto {mask_name} = {condition} >= 0.;
   {var_name} = {var_name}.blend({expr1}, {mask_name}).blend({expr2}, ~{mask_name});"""

    # Pattern 2: Comparison-based conditional (e.g., if v[10] > v[11])
    pattern2 = r'if\(\s*([^)]+)\s*>\s*([^)]+)\s*\)\s*{\s*([^=]+)=\s*([^;]+);\s*}\s*else\s*{\s*([^=]+)=\s*([^;]+);\s*}'

    def replace_comparison(match):
        left_expr = match.group(1).strip()
        right_expr = match.group(2).strip()
        target1 = match.group(3).strip()
        expr1 = match.group(4).strip()
        target2 = match.group(5).strip()
        expr2 = match.group(6).strip()

        var_name = target1
        global mask_number
        mask_name = f"mask_{mask_number}"
        mask_number += 1

        return f"""auto {mask_name} = {left_expr} > {right_expr};
   {var_name} = {var_name}.blend({expr1}, {mask_name}).blend({expr2}, ~{mask_name});"""

    # Pattern 3: Single conditional assignment
    pattern3 = r'if\(\s*([^)]+)\s*>=\s*0\.\s*\)\s*{\s*([^=]+)=\s*([^;]+);\s*}'

    def replace_single_conditional(match):
        condition = match.group(1).strip()
        target = match.group(2).strip()
        expr = match.group(3).strip()

        global mask_number
        mask_name = f"mask_{mask_number}"
        mask_number += 1

        return f"""auto {mask_name} = {condition} >= 0.;
   {target} = {target}.blend({expr}, {mask_name});"""

    # Apply transformations
    code = re.sub(pattern1,
                  replace_geq_zero,
                  code,
                  flags=re.MULTILINE | re.DOTALL)
    code = re.sub(pattern2,
                  replace_comparison,
                  code,
                  flags=re.MULTILINE | re.DOTALL)
    code = re.sub(pattern3,
                  replace_single_conditional,
                  code,
                  flags=re.MULTILINE | re.DOTALL)

    return code


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as f:
        code = f.read()

    result = convert_conditionals_to_blend(code)
    print(result)
