import logging

logger = logging.getLogger(__name__)

def standardize_skills_llm(value_dict, raw_client):
    tool_name = "standardize_header_name"

    prompt = (
        "You are a domain-aware naming standardizer for technical taxonomies.\n\n"
        "You will receive a dictionary where each key is a raw header proposal, and the value is a list of original tags matched to it (the array might be empty).\n\n"
        "Your task is:\n"
        "- Review if the header is already canonical, short, well-formatted, and well-known.\n"
        "- If yes, keep it as-is.\n"
        "- If no, suggest a standardized name that aligns with professional usage with its well known name (e.g. 'google bigquery', 'ci/cd').\n"
        "- If the header is not a skill or it is a soft skill, return empty string.\n"
        "Note that, you have to suggest a well defined standardized name.\n\n"
        "Return a dictionary where **each key is the raw header**, and **each value is a single best canonical string**.\n"
        "Do not modify the keys — only fix the values.\n"
        f"DICTIONARY OF VALUES: {value_dict}"
    )

    tool_schema = [{
        "name": tool_name,
        "description": (
            "Standardize raw headers based on their matched original tag variants based on stackoverflow.\n"
            "Return a dictionary where each key is the raw input header (as given), "
            "and the value is a single canonical string representing the standardized skill name.\n"
            "If the raw name is already canonical or it is a soft skill, return it as-is. "
        ),
        "input_schema": {
            "type": "object",
            "patternProperties": {
                "^[a-zA-Z0-9 _\\-]{3,50}$": {
                    "type": "string",
                    "description": "Your proposed tag to standardize the skill name.",
                }
            },
            "additionalProperties": False
        }
    }]

    try:
        logger.info(f"Sending {len(value_dict)} headers for standardization using model 'claude-3-7-sonnet-20250219'...")

        response = raw_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            messages=[{"role": "user", "content": prompt}],
            tools=tool_schema,
            max_tokens=2048,
            thinking={
                "type": "enabled",
                "budget_tokens": 1024
            },
            temperature=1,
            # tool_choice={"type": "tool", "name": tool_name}
        )

        # Directly access content[2].input as you confirmed
        result_dict = response.content[2].input

        if not isinstance(result_dict, dict):
            logger.error(f"Expected dictionary at response.content[2].input, got {type(result_dict)} instead.")
            return {k: k for k in value_dict.keys()}

        logger.info(f"Successfully standardized {len(result_dict)} headers.")
        return result_dict

    except IndexError:
        logger.error("response.content[2] is missing. Claude tool response might be malformed or incomplete.")
    except AttributeError as e:
        logger.error(f"Missing expected .input field in response.content[2]: {e}")
    except Exception as e:
        logger.exception("Unexpected error during standardization")

    # Fallback if anything goes wrong
    return {k: k for k in value_dict.keys()}


def skill_domain_classifier(skills, known_domains, raw_client):
    tool_name = "classify_skill_domain"

    known_domains = [
        "Software Engineering", "Data Science", "Machine Learning", "Deep Learning", "LLM", 
        "NLP", "Computer Vision", "Cloud Computing", "DevOps", "MLOps", "Data Engineering",
        "Cybersecurity", "Web Development", "Mobile Development", "Database",
        "Business Intelligence", "Project Management", "Computer Science", "Information Retrieval"
    ]

    # Prompt
    prompt = (
        "You are an expert in technical domains.\n\n"
        "You will receive a list of technical skills.\n\n"
        "Your job is:\n"
        "- For each skill, return a list of appropriate domains selected from the provided list.\n"
        "- Each assigned domain must be genuinely and independently recognized for that skill.\n"
        "- If none of the existing domains fit well, you may create one **new domain** that truly exists as a canonical technical field (similar to the provided domains).\n\n"
        "Important rules:\n"
        "- Only invent a new domain if absolutely necessary, and it must match real-world technical usage.\n"
        "- Prefer the narrowest, most specific domain available if both broad and narrow domains apply.\n"
        "- Avoid assigning domains based only on surface similarity; base it on technical usage and community standards.\n"
        "- Your output must be a JSON dictionary where each key is the skill, and the value is the assigned domain(s).\n\n"
        f"Skills to classify:\n{skills}\n\n"
        "List of allowed domains:\n"
        f"{', '.join(known_domains)}\n\n"
    )

    # Tool schema
    tool_schema = [{
        "name": tool_name,
        "description": "Classify each skill into a technical domain or invent a new domain if necessary.",
        "input_schema": {
            "type": "object",
            "patternProperties": {
                "^[a-zA-Z0-9 ()\\-/]{2,100}$": {
                    "type": "array",
                    "description": "The assigned domain name for this skill.",
                    "items": {
                        "type": "string",
                    },
                }
            },
            "additionalProperties": False
        }
    }]

    # try:
    # print(f"Sending {len(skills)} skills for domain classification...")

    try:

        response = raw_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            messages=[{"role": "user", "content": prompt}],
            tools=tool_schema,
            max_tokens=2048,
            thinking={
                "type": "enabled",
                "budget_tokens": 1024
            },
            temperature=1,
            # tool_choice={"type": "tool", "name": tool_name}
        )

        # Directly access content[2].input as you confirmed
        result_dict = response.content[2].input

        if not isinstance(result_dict, dict):
            logger.error(f"Expected dictionary at response.content[2].input, got {type(result_dict)} instead.")
            return {k: [] for k in skills}

        logger.info(f"Successfully standardized {len(result_dict)} headers.")
        return result_dict

    except IndexError:
        logger.error("response.content[2] is missing. Claude tool response might be malformed or incomplete.")
    except AttributeError as e:
        logger.error(f"Missing expected .input field in response.content[2]: {e}")
    except Exception as e:
        logger.exception("Unexpected error during standardization")

    # Fallback if anything goes wrong
    return {k: [] for k in skills}    


def classify_subdomains(skills_to_domain, raw_client):
    """
    Classify subdomains based on the provided domain and skills mapping.

    Args:
        skills_to_domain (dict): Dictionary of domains and their associated skills.
        raw_client: Client for making API calls.

    Returns:
        dict: Dictionary of domains and their classified subdomains.
    """
    
    # Tool name
    tool_name = "classify_subdomains"

    prompt = (
        "You are a non-technical recruiter structuring skills into subdomains.\n\n"
        "Subdomains are meant to be in a 'skills' section of a resume as header for a set of skills. "
        "Given a dictionary of domains and skills under it, create subdomains that covers the skills and places under a domain. "
        "In other words, subdomains place between skills and domains.\n"
        "RULES:\n"
        "1- In addition, look at each domain and skills under it independently like other domains-skills mapping do not exist.\n"
        "2- Be critical about subdomains naming to be short, and meaningful under the domain context.\n"
        "3- If a skill under a domain is not a popular choice (is a stretch), you can ignore it. "
        "4- If there are subdomains for a domain we don't have skillset for it yet, do not hesitate to add.\n"
        "Thought process:\n"
        "1- Can domain be divided into: tools, frameworks, libraries, models, techniques, etc.?\n"
        "2- What are the most popular terminologies for subdomains in the industrial job descriptions?\n"
        "3- Are the chosen subdomains short and non-technical?\n"
        "4- Can it be used as the header of skills in a resume for a job?\n"
        "Output a valid JSON where each key is a domain and value is an array of subdomains:\n"
        "EXAMPLE:\n"
        "{Machin Learning: [ML models, ML frameworks, ML libraries, ML tools]}\n\n"
        f"{skills_to_domain}"

    )

    # Tool schema
    tool_schema = [{
        "name": tool_name,
        "description": "Suggest clean, technically valid subdomains for each main domain.",
        "input_schema": {
            "type": "object",
            "patternProperties": {
                "^[a-zA-Z0-9 ()\\-/]{2,100}$": {
                    "type": "array",
                    "description": "List of clean subdomains for this domain.",
                    "items": {
                        "type": "string",
                    },
                }
            },
            "additionalProperties": False
        }
    }]

    response = raw_client.messages.create(
        model="claude-3-7-sonnet-20250219",
        messages=[{"role": "user", "content": prompt}],
        tools=tool_schema,
        max_tokens=4096,
        thinking={
            "type": "enabled",
            "budget_tokens": 2048
        },
        temperature=1,
        # tool_choice={"type": "tool", "name": tool_name}
    )
    
    return response.content[2].input


def subdomain_skill_mapping(domain: str, skills: list, known_subdomains: list, raw_client) -> dict:
    """
    Classify a list of skills into appropriate subdomains using an LLM.
    
    Args:
        skills (list): List of skills to classify.
        known_subdomains (list): List of allowed subdomains.
        raw_client: The LLM client to use for classification.
        
    Returns:
        dict: Mapping from subdomain -> list of skills.
    """
    tool_name = "classify_skill_subdomain"

    # Prompt
    prompt = (
        "You are an expert in technical skill structuring.\n\n"
        f"You will receive a list of subdomains in domain '{domain}' along with a list of technical skills.\n\n"
        "Your job is:\n"
        "- For each subdomain, pick all appropriate skills that can fall under the subdomain umbrella.\n"
        "- Each skill can go under multiple subdomains, as far as they are technically and DIRECTLY related and known for it.\n"
        "- Each assigned skill must be genuinely and independently recognized in that subdomain.\n\n"
        "Important rules:\n"
        "- Avoid assigning skills based only on keyword matching; base it on technical meaning and professional usage.\n"
        "- Make sure it makes sense to assign the skill to the subdomain.\n"
        "- If a subdomain has no skills, you should return an empty list for it.\n\n"
        "Your output must be a JSON dictionary where each KEY is the SUBDOMAIN, and the VALUE is a LIST of assigned SKILLS.\n\n"
        "OUTPUT FORMAT EXAMPLE:\n"
        "{'Subdomain1': ['Skill1', 'Skill2'], 'Subdomain2': ['Skill2', 'Skill5']}\n\n"
        f"List of Skills:\n{skills}\n\n"
        "List of subdomains:\n"
        f"{', '.join(known_subdomains)}\n\n"
    )

    # Tool schema
    tool_schema = [{
        "name": tool_name,
        "description": "For each subdomain finding a list of related skills.",
        "input_schema": {
            "type": "object",
            "patternProperties": {
                "^[a-zA-Z0-9 ()\\-/]{2,100}$": {
                    "type": "array",
                    "description": "The list of skills assigned to this subdomain.",
                    "items": {
                        "type": "string",
                    },
                }
            },
            "additionalProperties": False
        }
    }]

    # LLM Call
    response = raw_client.messages.create(
        model="claude-3-7-sonnet-20250219",
        messages=[{"role": "user", "content": prompt}],
        tools=tool_schema,
        max_tokens=2048,
        thinking={
            "type": "enabled",
            "budget_tokens": 1024
        },
        temperature=1,
        # tool_choice={"type": "tool", "name": tool_name}
    )

    return response.content[2].input  # You will likely extract the tool result from response later
