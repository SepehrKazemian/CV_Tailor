from typing import List, Dict
from pydantic import BaseModel
import json
from resume_tailor.utils.pydantic_to_schema import pydantic_model_to_schema
from resume_tailor.tailors.skills.skills_prompts import (
    skill_extractor_system_prompt,
    skill_map_system_prompt
    )
from langchain.vectorstores.base import VectorStoreRetriever
from resume_tailor.tailors.skills import schema as sch
from importlib import reload
import ast

def single_class_skill_extraction(raw_client, model_name: str, job_description: str, class_name: str, class_description: str) -> List[str]:
    """
    Extract a single class of skills from the job description using the given class name and description.
    """
    reload(sch)
    prompt = (
        f"You are a resume skill extraction engine.\n\n"
        f"From the following job description, extract ALL the keywords belong to the category '{class_name}'.\n\n"
        f"Definition of '{class_name}': {class_description}\n\n"
        f"Guidelines:\n"
        f"- Only include **individual, clearly named technologies**: tools, libraries, programming languages, platforms, frameworks.\n"
        f"- DO split grouped items.\n"
        f"- DO NOT include grouped phrases or academic degrees.\n"
        f"- DO NOT miss any keyword skill. It is mandatory for ATS. If you're hesitant, add it to the list.\n"
        f"Return the result as a **valid JSON array** of strings like this: [\"Python\", \"Docker\"]\n\n"
        f"Job Description:\n{job_description}"
    )
    
    CLASS_MODEL_MAP = {
        "Primary": sch.PrimarySkillMap,
        "Inferred": sch.InferredSkillMap,
        "Inferred_Secondary": sch.InferredSecondarySkillMap,
    }

    tool_schema = [{
        "name": "skill_finder",
        "description": f"From job description find all of the {CLASS_MODEL_MAP[class_name]}.  Return it in an array.",
        "input_schema": pydantic_model_to_schema(CLASS_MODEL_MAP[class_name]),
    }]

    response = raw_client.messages.create(
        model=model_name,
        tools=tool_schema,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024,
        tool_choice={"type": "tool", "name": "skill_finder"}
    )

    print(response.content[0].input)
    return response.content[0].input[class_name]

def scored_skill_extractor(raw_client, model_name: str, job_description: str) -> Dict[str, List[str]]:
    """
    Extracts relevant skills from a job description in four separate LLM calls for each skill class (Class_5 to Class_2).
    """
    class_map = {
        "Primary": "Skills explicitly mentioned in job description. These are critical, must-have, or preferred skills for the role. Don't miss any keyword.",
        "Inferred": "Skills that are not explicitly stated in the job description, but are logically implied by the core responsibilities or tech stack. These support the primary skills and reflect domain-relevant experience. These skills MUST NOT be in the job description.",
        "Inferred_Secondary": "Niche, fringe, or advanced tools/skills that are not mentioned in the job description but may be useful. These are weakly inferred and contextually relevant in edge cases. These skills MUST NOT be in the job description."
    }
    

    skill_map = {}
    for class_name, description in class_map.items():
        skills = single_class_skill_extraction(raw_client, model_name, job_description, class_name, description)
        if isinstance(skills, str):
            skills = ast.literal_eval(skills)
        skill_map[class_name] = skills

    return skill_map