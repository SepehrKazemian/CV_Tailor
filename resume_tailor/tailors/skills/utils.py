import re
from typing import List, Dict, Tuple


def sanitize_skill(skill: str) -> str:
    """
    Sanitizes a skill string by replacing any sequence of non-alphanumeric characters
    with an underscore, and stripping leading/trailing underscores.
    """
    sanitized = re.sub(r"[^0-9a-zA-Z]+", "_", skill).strip("_")
    return sanitized


# Process candidate skills and missing skills, building mappings.
def sanitize_skills_with_mapping(skills: List[str]) -> (List[str], Dict[str, str]):
    """
    Returns a tuple:
      - A list of sanitized skill strings.
      - A mapping dictionary that maps the sanitized skill back to the original.
    """
    sanitized_list = []
    mapping = {}
    for skill in skills:
        sanitized = sanitize_skill(skill)
        sanitized_list.append(sanitized)
        mapping[sanitized] = skill
    return sanitized_list, mapping

def sanitize_headers_with_mapping(
    skills_dict: Dict[str, List[str]]
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Sanitizes only the dictionary keys (headers), preserving skill lists.

    Returns:
      - A new dict: {sanitized_header: original_skills}
      - A header mapping: {sanitized_header: original_header}
    """
    sanitized_dict = {}
    header_mapping = {}

    for header, skills in skills_dict.items():
        sanitized_header = sanitize_skill(header)
        sanitized_dict[sanitized_header] = skills
        header_mapping[sanitized_header] = header

    return sanitized_dict, header_mapping
