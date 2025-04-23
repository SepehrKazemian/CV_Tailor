

from typing import List
import json

def comprehensive_skill_grouping(raw_client, model_name, nodes_str):
    tool_name = "comprehensive_skill_clustering"
    prompt = (
        f"You are given a flat list of individual technical skills extracted from a resume:\n\n{nodes_str}\n\n"
        "Your task is to group these skills into logical clusters that would appear as section headers in a resume.\n\n"
        "IMPORTANT: "
        "1. Each skill can and should appear in MULTIPLE clusters if it fits in more than one category. "
        "For example, 'SQL' could appear in both 'Database Technologies' and 'Data Analysis'.\n\n"
        "2. A skill can ALSO serve as a category header if it represents a broader domain. For example, "
        "'Data Science' could be both a skill under 'Analytics' AND a category header "
        "that contains skills like 'Statistics' and 'Predictive Modeling'.\n\n"
        "Each cluster should represent a broader category or resume header (e.g., 'Cloud Computing', 'UX/UI Design', 'Backend Development') "
        "and contain a list of related subskills that naturally belong together.\n\n"
        "Return the result **directly** as a dictionary, where each key is a cluster name and the value is an array of subskills grouped under that category.\n\n"
        "Be thorough and comprehensive - consider ALL possible ways a skill could be categorized in a resume context.\n\n"
        "Be consistent and concise with cluster names. Use professional, resume-appropriate section headers.\n"
        "Return your result as a **pure JSON object**. Each value must be a JSON array."
        "Do not wrap the output in any top-level key. \n\n"
        "Example input:\n['SQL', 'Excel', 'Tableau', 'Data Science', 'Statistics', 'R', 'Git']\n\n"
        "Example output:\n{\n"
        "  'Database Technologies': ['SQL'],\n"
        "  'Data Analysis': ['SQL', 'Excel', 'R', 'Statistics'],\n"
        "  'Data Visualization': ['Tableau', 'Excel'],\n"
        "  'Analytics': ['Data Science', 'R', 'Statistics'],\n"
        "  'Data Science': ['Statistics', 'R'],\n"
        "  'Version Control': ['Git']\n"
        "}\n\n"
    )

    desc = (
        "Given a flat list of resume skills, group them into comprehensive logical clusters for a professional resume. "
        "Each skill should appear in ALL relevant categories. Additionally, a skill can ALSO serve as a category header "
        "if it represents a broader domain (e.g., 'Data Science' could be both a skill and a category header). "
        "Create all reasonable categorizations that would make sense in a professional resume context."
    )

    schema = {
        "type": "object",
        "patternProperties": {
            "^[a-zA-Z0-9 _\\-]{3,50}$": {
                "type": "array",
                "description": "A list of related skills that belong in this cluster. Skills can appear in multiple clusters.",
                "items": {
                    "description": "A skill that belongs in this category.",
                    "type": "string"
                },
            },
        },
        "additionalProperties": False,
    }

    tool_schema = [{
        "name": tool_name,
        "description": desc,
        "input_schema": schema
    }]

    response = raw_client.messages.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        tools=tool_schema,
        max_tokens=2048,
        temperature=0.1,
        tool_choice={"type": "tool", "name": tool_name}
    )
    
    return response.content[0].input

def grouping_skills(raw_client, model_name, nodes_str):
    tool_name = "skill_clustering"
    prompt = (
        f"You are given a flat list of individual technical skills extracted from a resume:\n\n{nodes_str}\n\n"
        "Your task is to group these skills into logical clusters that would appear as section headers in a resume.\n\n"
        "Each cluster should represent a broader category or resume header (e.g., 'DevOps Tools', 'Data Visualization', 'Web Technologies') "
        "and contain a list of related subskills that naturally belong together.\n\n"
        "Return the result **directly** as a dictionary, where each key is a cluster name and the value is an array of subskills grouped under that category.\n\n"
        "If a skill is a standalone concept and does not logically belong to another group, and represents a broad or umbrella concept, "
        "you may return it as a cluster with an empty list.\n\n"
        "Be consistent and concise with cluster names. Avoid vague labels like 'Miscellaneous' or 'General'."
        "Return your result as a **pure JSON object**. Each value must be a JSON array."
        "Do not wrap the output in any top-level key. \n\n"
        "Example input:\n['Kubernetes', 'Grafana', 'D3.js', 'Jenkins', 'Data Science']\n\n"
        "Example output:\n{\n  'DevOps Tools': ['Kubernetes', 'Jenkins'],\n  'Data Visualization': ['Grafana', 'D3.js'],\n  'Data Science': []\n}\n\n"
    )

    desc = (
        "Given a flat list of resume skills, group them into logical clusters that can be used as section headers "
        "in a resume. Each cluster should be a meaningful category (e.g., 'Data Visualization', 'Cloud Platforms', etc.) "
        "with a list of related subskills. If a skill represents a broad domain or umbrella concept, leave the cluster empty."
    )

    schema = {
        "type": "object",
        "patternProperties": {
            "^[a-zA-Z0-9 _\\-]{3,50}$": {
                "type": "array",
                "description": "A list of related skills that belong in this cluster.",
                "items": {
                    "description": "The output should be a dictionary where each key is a cluster name and its value is a list of related skills.",
                    "type": "string"
                },
            },
        },
        "additionalProperties": False,
    }

    tool_schema = [{
        "name": tool_name,
        "description": desc,
        "input_schema": schema
    }]

    response = raw_client.messages.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        tools=tool_schema,  # Passing our tool schema here.
        max_tokens=1024,
        temperature=0.1,
        tool_choice={"type": "tool", "name": tool_name}
    )
    
    return response.content[0].input


def suggest_possible_section_combinations(raw_client, model_name, header_nodes: List[str]):
    tool_name = "header_section_combinator"
    nodes_str = json.dumps(header_nodes, indent=2)

    prompt = (
        f"You are given a list of resume skill section headers:\n\n{nodes_str}\n\n"
        "Your task is to suggest all reasonable **combinations** of two or more headers that can logically be merged "
        "into a broader resume section name.\n\n"
        "Each combination should result in a new group name, and the value should be the list of original section headers it combines.\n\n"
        "Only combine headers that share a clear semantic or technical relationship. Avoid combining unrelated or overly generic headers.\n\n"
        "Example input:\n['Natural Language Processing', 'Information Retrieval', 'Computer Vision', 'Statistical Modeling']\n\n"
        "Example output:\n{\n  'NLP & IR': ['Natural Language Processing', 'Information Retrieval'],\n"
        "  'AI Domains': ['Natural Language Processing', 'Computer Vision'],\n"
        "  'Modeling & Analytics': ['Statistical Modeling', 'Information Retrieval']\n}\n\n"
        "Do not return standalone headers. Only return merged ones.\n"
        "Do not wrap the result in a top-level key like 'skills'.\n"
        "Output strictly as a flat JSON dictionary."
    )

    desc = (
        "Given a list of resume section headers, generate a dictionary where each key is a possible merged header "
        "(e.g., 'NLP & IR'), and its value is the list of original section headers it merges. "
        "Only suggest combinations that are reasonable and meaningful. Do not include standalone headers."
    )

    schema = {
        "type": "object",
        "patternProperties": {
            "^[a-zA-Z0-9 &\\-]{3,50}$": {
                "type": "array",
                "description": "List of original section headers merged into this group.",
                "minItems": 2,
                "items": {
                    "type": "string"
                },
            }
        },
        "additionalProperties": False,
    }

    tool_schema = [{
        "name": tool_name,
        "description": desc,
        "input_schema": schema
    }]

    response = raw_client.messages.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        tools=tool_schema,
        max_tokens=1024,
        temperature=0.2,
        tool_choice={"type": "tool", "name": tool_name}
    )

    return response.content[0].input



def grouping_skill_sections(raw_client, model_name, header_nodes: List[str]):
    tool_name = "header_section_clustering"
    nodes_str = json.dumps(header_nodes, indent=2)

    prompt = (
        f"You are given a list of skill section headers extracted from a resume:\n\n{nodes_str}\n\n"
        "Your task is to group these headers into broader, consolidated resume skill sections.\n\n"
        "Each output key should represent a new section name that combines similar headers (e.g., 'Cloud & DevOps', 'ML/AI Stack'), "
        "and its value should be a list of the original section headers that belong in that group.\n\n"
        "If a section header represents a self-contained concept, you may return it as its own group with an empty list.\n\n"
        "Example input:\n['Data Analytics', 'Visualization Tools', 'Containerization', 'Orchestration']\n\n"
        "Example output:\n{\n  'Data & Visualization': ['Data Analytics', 'Visualization Tools'],\n  'Infrastructure': ['Containerization', 'Orchestration']\n}\n\n"
        "Avoid vague names like 'General Skills'. Be clear and concise."
        "Do not wrap the output in any top-level key."
    )

    desc = (
        "Group related resume skill section headers into broader, combined clusters. "
        "The goal is to reduce fragmentation and make the resume layout cleaner. Each key in the final JSON "
        "should be a new merged section name, and the value is an array of original headers it represents. "
        "Leave a cluster empty if it's a valid standalone concept."
    )

    schema = {
        "type": "object",
        "patternProperties": {
            "^[a-zA-Z0-9 _\\-]{3,50}$": {
                "type": "array",
                "description": "The output is a dictionary where each key is a new merged section name, and its value is a list of section headers grouped under it.",
                "items": {
                    "type": "string",
                    "description": "List of original section headers grouped under this new section."
                },
            }
        },
        "additionalProperties": False,
    }

    tool_schema = [{
        "name": tool_name,
        "description": desc,
        "input_schema": schema
    }]

    response = raw_client.messages.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        tools=tool_schema,
        max_tokens=1024,
        temperature=0.1,
        tool_choice={"type": "tool", "name": tool_name}
    )
    
    return response.content[0].input