import json

def new_skills_to_all_headers(raw_client, model_name, skills, headers):
    tool_name = "header_skill_mapper"

    # Step 1: Format the list of skills
    skills_str = json.dumps(skills, indent=2)

    # Step 2: Prompt (headers are NOT mentioned here!)
    prompt = (
        "You are organizing a list of skills into appropriate resume section headers.\n\n"
        "Given a list of skills extracted from a resume "
        "Group these skills under logical resume section headers using the provided tool.\n"
        "Your task is to independently assign relevant skills to **each header**, one at a time.\n"
        "Do NOT compare headers against each other."
        "Each header is processed in isolation — just find what fits."
        "Return your answer as a valid JSON object, where each key is a header name "
        "and its value is a list of matching skills.\n\n"
        "List of Skills:\n"
        f"{skills_str}\n\n"
    )

    # Step 3: Schema
    schema = {
        "type": "object",
        "description": "Map each header to a list of skills that belong under that category. A skill can belong to multiple headers. Omit any skill that do not clearly belong under a given header or are too vague or general.",
        "properties": {
            header: {
                "type": "array",
                "items": {"type": "string"}
            } for header in headers
        },
        "additionalProperties": False,
        "required": headers
    }

    tool_schema = [{
        "name": tool_name,
        "description": (
            "Assign each skill to at least one of the predefined resume section headers. "
            "Each skill can appear under multiple headers. Omit any skill that do not clearly belong under a given header or are too vague or general."
        ),
        "input_schema": schema
    }]

    # Step 4: Call LLM
    response = raw_client.messages.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        tools=tool_schema,
        max_tokens=1024,
        temperature=0,
        tool_choice={"type": "tool", "name": tool_name}
    )
    
    return response.content[0].input


def judge_mapping_feedback_with_score(
    raw_client,
    model_name: str,
    skills_header_mapping: dict[str, list[str]],
    previous_score: int,
    temperature: float = 0,
):
    """
    Uses an LLM to critique a skill-to-header mapping and return structured feedback
    along with a numeric quality score (0–100). This is useful for resume quality audits.
    """

    tool_name = "skill_header_feedback"

    prompt = (
        "You are a serious but fair resume sectioning reviewer who only points to issues if any exists.\n\n"
        "You will receive a JSON mapping of resume section headers to their associated skills.\n"
        "Your task is to:\n"
        "1. Analyze and critique the mapping.\n"
        "2. Highlight only issues such as:\n"
        "   - Skills that are too vague or generic.\n"
        "   - Misclassified or out-of-place skills or duplicates.\n"
        "       - One skill is allowed to be under multiple headers.\n"
        "   - Skills that belong to multiple classes but they are only in some of them.\n"
        "3. Provide a quality grade (0–100) for how good this mapping is for a professional resume.\n\n"
        "DO NOT suggest removing any header. \n"
        "Grade Guidelines:\n"
        "- 90–100 = excellent (can show to CEO of Google)\n"
        "- 70–89  = solid, but some issues remain\n"
        "- 50–69  = usable, but needs review\n"
        "- Below 50 = poor; rework required\n"
        f"It is a fixed version of previous feedback of yours. It was scored: {previous_score}/100.\n"
        f"Here is the current mapping:\n{json.dumps(skills_header_mapping, indent=2)}"
    )

    schema = {
        "type": "object",
        "properties": {
            "feedback": {
                "type": "string",
                "description": "Natural language critique of the skill-to-header mapping."
            },
            "grade": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "A numeric score from 0 to 100 evaluating the overall quality of the classification."
            }
        },
        "required": ["feedback", "grade"],
        "additionalProperties": False
    }

    tool_schema = [{
        "name": tool_name,
        "description": "Reviews a resume skill-to-header mapping and returns critique plus quality grade.",
        "input_schema": schema
    }]

    response = raw_client.messages.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        tools=tool_schema,
        temperature=temperature,
        max_tokens=1024,
        tool_choice={"type": "tool", "name": tool_name}
    )

    return response.content[0].input


def reclassify(
    raw_client,
    model_name: str,
    skills_header_mapping: list[str],
    feedback = str,
    temperature: float = 0,
) -> dict[str, list[str]]:
    """
    Uses a simple chat‐style prompt (no tools) to classify each skill under one or 
    more of the provided headers. Returns a dict: { header: [skills…], … }.
    """
    tool_name = "schema_judge_llm"
    prompt = (
            "You are an expert resume skills auditor.\n\n"
            "You are given:\n"
            "1. A skill-to-header mapping (JSON object)\n"
            "2. Feedback about potential classification issues\n\n"
            "Your task is to return a **cleaned version** of the mapping following these strict rules:\n"
            "• Do NOT add new skills unless strongly implied (avoid hallucination)\n"
            "• Do NOT remove any headers — all headers from the input MUST be present in the output (even if empty)\n"
            "• Every value must be a **JSON array** of strings\n"
            "• If a header ends up empty, try your best to add relevant skills, otherwise return it with an empty array\n"
            "Return your answer as a valid JSON object, where each key is a header name "
            "and its value is a list (array) of skills.\n\n"
            f"Feedback:\n{feedback}\n\n"
            f"Skill-to-Header Mapping:\n{json.dumps(skills_header_mapping, indent=2)}"
        )
    
    
    schema = {
        "type": "object",
        "patternProperties": {
            "^[a-zA-Z0-9 _\\-]{3,50}$": {
                "type": "array",
                "description": "A JSON of headers with an array of relevant skills to each header.",
                "items": {
                    "type": "string",
                    "description": "An array of related skill for the category.",
                },
            }
        },
        "additionalProperties": False,
    }

    desc ="Clean up a skill-to-header mapping based on feedback."
    
    tool_schema = [{
        "name": tool_name,
        "description": desc,
        "input_schema": schema
    }]
    
    response = raw_client.messages.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        tools=tool_schema,
        temperature=0,
        max_tokens=2048,
        tool_choice={"type": "tool", "name": tool_name}
    )

    # the assistant message should be pure JSON
    
    return response.content[0].input

def judge_mapping_feedback_with_score(
    raw_client,
    model_name: str,
    skills_header_mapping: dict[str, list[str]],
    temperature: float = 0.0,
):
    """
    Uses an LLM to critique a skill-to-header mapping and return structured feedback
    along with a numeric quality score (0–100). This is useful for resume quality audits.
    """

    tool_name = "skill_header_feedback"

    prompt = (
        "You are a resume sectioning reviewer.\n\n"
        "You will receive a JSON mapping of resume section headers to their associated skills.\n"
        "Your task is to:\n"
        "1- The goal of final mapping is every skill to show up under all appropriate classes. \n"
        "2. Highlight only issues below:\n"
        "   - Skills that are too vague or generic.\n"
        "   - Misclassified or out-of-place skills.\n"
        "3. Provide a quality grade (0–100) for how good this mapping is for a professional resume.\n\n"
        "Grade Guidelines:\n"
        "- 90–100 = excellent (can show to CEO of Google)\n"
        "- 70–89  = solid, but some issues remain\n"
        "- 50–69  = usable, but needs review\n"
        "- Below 50 = poor; rework required\n"
        "CRITICAL RULES:"
        "   - NEVER mention where  ."
        f"Here is the current mapping:\n{json.dumps(skills_header_mapping, indent=2)}"
    )

    schema = {
        "type": "object",
        "properties": {
            "feedback": {
                "type": "string",
                "description": "Natural language critique of the skill-to-header mapping."
            },
            "grade": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "A numeric score from 0 to 100 evaluating the overall quality of the classification."
            }
        },
        "required": ["feedback", "grade"],
        "additionalProperties": False
    }

    tool_schema = [{
        "name": tool_name,
        "description": "Reviews a resume skill-to-header mapping and returns critique plus quality grade.",
        "input_schema": schema
    }]

    response = raw_client.messages.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        tools=tool_schema,
        temperature=temperature,
        max_tokens=1024,
        tool_choice={"type": "tool", "name": tool_name}
    )

    return response.content[0].input



def skill_mapping(
    raw_client,
    model_name: str,
    header: list[str],
    skills = str,
    temperature: float = 0,
) -> dict[str, list[str]]:
    """
    Uses a simple chat‐style prompt (no tools) to classify each skill under one or 
    more of the provided headers. Returns a dict: { header: [skills…], … }.
    """
    tool_name = "mapping_review"
    prompt = (
            "You are an expert in resume editor.\n"
            f"Based on Skills, return an array of skills that can be under header {header} in a professional CV.\n"
            f"Skills:\n{skills}\n\n"
        )
    
        
    schema = {
        "type": "object",
        "description": f"Select relevant skills for header '{header}' in the skills section of a CV.",
        "properties": {
            "skills": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "additionalProperties": False,
        "required": ["skills"]
    }

    
    tool_schema = [{
        "name": tool_name,
        "description": "Select appropriate skills under umbrella of header",
        "input_schema": schema
    }]
    
    response = raw_client.messages.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        tools=tool_schema,
        temperature=0,
        max_tokens=512,
        tool_choice={"type": "tool", "name": tool_name}
    )

    # the assistant message should be pure JSON
    
    return response.content[0].input
