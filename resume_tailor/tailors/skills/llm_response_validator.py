import ast

def validate_and_fix_skill_mapping(mapping: dict) -> dict:
    """
    Validates and fixes a skill-header mapping.
    - Ensures each value is a list.
    - If it's a string, tries to parse it using ast.literal_eval.
    - Raises ValueError on failure.

    Args:
        mapping (dict): The raw skill-to-header JSON object.

    Returns:
        dict: A corrected mapping with all values as lists.
    """
    fixed_mapping = {}

    for header, value in mapping.items():
        if isinstance(value, list):
            fixed_mapping[header] = value

        elif isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    fixed_mapping[header] = parsed
                else:
                    return {}
            except (SyntaxError, ValueError) as e:
                return {}

        else:
            return {}

    return fixed_mapping
