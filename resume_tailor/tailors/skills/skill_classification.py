from typing import List, Dict, Any
import json
import re
import ast


def generate_missing_skills_tool_schema(missing_skills: List[str],
                                        tool_name: str = "missing_skills_checker") -> List[Dict[str, Any]]:
    """
    Given a list of missing JD skills, generates a JSON schema for an LLM tool.
    Each missing skill will be a property (key) of type boolean.
    The returned schema is wrapped in a list with one dictionary containing keys:
       - name: The tool name.
       - description: A description of the tool.
       - input_schema: The JSON schema (object with properties for each missing skill).
    """
    properties: Dict[str, Any] = {}
    for skill in missing_skills:
        properties[skill] = {
            "type": "number",
            "minimum": 0,
            "maximum": 100,            
            "description": f"Likelihood score for skill '{skill}'."
        }
    
    schema = {
        "type": "object",
        "properties": properties,
        "required": missing_skills
    }
    
    desc = (
        "You are estimating how likely a candidate is to know each of the given missing skills, "
        "based on their known technical skills and tools.\n\n"
        "Consider real-world co-occurrence, common tool stacks, domain overlap, and conceptual relatedness.\n\n"
        "Use this 0–100 scale:\n"
        "- 100 = Very likely (frequently seen together or clearly implied)\n"
        "- 70 = Likely (commonly associated or complementary)\n"
        "- 50 = Possible (some overlap, weakly related)\n"
        "- 0 = Very unlikely (unrelated, uncommon pairing)\n\n"
        "Be analytical — do not assume skills are known unless there is a strong reason."
        "Return a clean JSON object matching each skill to a number between 0 and 100."
    )    
    tool_schema = [{
        "name": tool_name,
        "description": desc,
        "input_schema": schema
    }]
    
    return tool_schema

def sanitize_skill(skill: str) -> str:
    """
    Sanitizes a skill string by replacing any sequence of non-alphanumeric characters
    with an underscore, and stripping leading/trailing underscores.
    """
    sanitized = re.sub(r'[^0-9a-zA-Z]+', '_', skill).strip('_')
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


def skill_matching_possibility(skill_classification, ):
    # Combine missing skills for "Primary" and "Inferred" JD classes.
    missing_primary = skill_classification.get("missing_skills_report", {}).get("Primary", [])
    missing_inferred = skill_classification.get("missing_skills_report", {}).get("Inferred", [])
    combined_missing = list(set(missing_primary + missing_inferred))

    # Build a prompt for the LLM.
    # Here we provide context by including the candidate skills from the "Primary" and "Inferred" buckets.
    candidate_primary = skill_classification.get("candidate_primary", [])
    candidate_inferred = skill_classification.get("candidate_inferred", [])
    combined_candidates = list(set(candidate_primary).union(set(candidate_inferred)))

    # Generate the tool input schema based on combined missing skills.
    tool_schema = generate_missing_skills_tool_schema(combined_missing)


    # Sanitize candidate skills.
    sanitized_candidates, candidate_mapping = sanitize_skills_with_mapping(combined_candidates)
    # Sanitize missing skills.
    sanitized_missing, missing_mapping = sanitize_skills_with_mapping(combined_missing)

    # Generate the tool input schema based on the sanitized missing skills.
    tool_schema = generate_missing_skills_tool_schema(sanitized_missing)


    prompt = (
        "You are a candidate skill resolver.\n\n"
        "The candidate is known to possess the following technical skills:\n"
        f"{json.dumps(combined_candidates)}\n\n"
        "We are now evaluating a list of other skills (called 'missing skills') to determine "
        "how likely it is that the candidate also knows them, based on the known skills.\n\n"
        "Your task is to return a number from 0 to 100 for each missing skill, indicating how likely the candidate is to know it:\n"
        "- 100 = extremely likely (the missing skill is clearly related or implied)\n"
        "- 50 = somewhat likely (may be partially related or commonly co-occurring)\n"
        "- 0 = very unlikely (no known skills relate to it)\n\n"
        "Use the structured tool provided to return your answer as JSON. Only include the required keys from the schema. "
        "Do not include explanations — only the JSON output."
    )

    response = raw_client.messages.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        tools=tool_schema,  # Passing our tool schema here.
        max_tokens=1024,
        temperature=0.1,
        tool_choice={"type": "tool", "name": "missing_skills_checker"}
    )