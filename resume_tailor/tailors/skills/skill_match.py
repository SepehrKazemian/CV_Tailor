from typing import List, Dict, Set
from pydantic import BaseModel
import json
from resume_tailor.utils.pydantic_to_schema import pydantic_model_to_schema
from resume_tailor.tailors.skills.skills_prompts import (
    skill_extractor_system_prompt,
    skill_map_system_prompt
    )
from langchain.vectorstores.base import VectorStoreRetriever
from resume_tailor.tailors.skills.schema import SkillExtractionOutput, SkillMapOutput
import numpy as np

def skill_extractor(raw_client, model_name: str, job_description: str) -> List[str]:
    """
    Extracts relevant skills from a given job description. The result should be passed to skill matcher to match with candidate's skills.
    """    
    prompt = (
        f"You are a resume skill extraction engine.\n\n"
        f"Find and return all of the skills mentioned the job posting as an array .\n\n"
        f"Guidelines:\n"
        f"- Only include **individual, clearly named technologies**: tools, libraries, programming languages, platforms, frameworks.\n"
        f"- DO split grouped items.\n"
        f"- DO NOT include grouped phrases or academic degrees.\n"
        f"- DO NOT miss any keyword skill. It is mandatory for ATS. If you're hesitant, add it to the list.\n"
        f"Return the result as a **valid JSON array** of strings like this: [\"C#\", \"GoLang\"]\n\n"
        f"Job Description:\n{job_description}"
    )
    tool_schema = [{
        "name": "skill_finder",
        "description": "Find all of the skills in the job description and return it in a list.",
        "input_schema": pydantic_model_to_schema(SkillExtractionOutput),
    }]

    response = raw_client.messages.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        tools=tool_schema,
        max_tokens=1024,
        tool_choice={"type": "tool", "name": "skill_finder"},
    )
    return response.content[0].input



def final_out(raw_client, model_name:str, matched_skills: List[str]) -> Dict[str, List[str]]:
    """
    Organizes matched skills into logical headers and returns the final skill map.
    """
    tool_schema = [{
        "name": "return_skill_map",
        "description": "Return the final mapping of matched job-relevant skills grouped by topic headers.",
        "input_schema": pydantic_model_to_schema(SkillMapOutput),
    }]

    response = raw_client.messages.create(
        model=model_name,
        system=skill_map_system_prompt,
        messages=[
            {"role": "user", "content": f"Organize the following skills into logical groups and return a dictionary:\n\n{json.dumps(matched_skills)}"},
        ],
        tools=tool_schema,
        max_tokens=1024,
        tool_choice={"type": "tool", "name": "return_skill_map"},
    )
    return response.content[0].input["skill_map"]
