
from typing import Dict, List, Any
from resume_tailor.utils.pydantic_to_schema import pydantic_model_to_schema
from resume_tailor.tailors.tailor_utils import SKILL_SECTION_HEADERS
from resume_tailor.tailors.skills.schema import HeaderSkillsComparison
import re
import json

def sanitize_header(header: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', '_', header)[:64]


class SkillMatcher:
    def __init__(self, llm, logger, model_name):
        self.client = llm
        self.logger = logger
        self.headers = SKILL_SECTION_HEADERS
        self.model_name = model_name
        self.map_header = {}
    
    def get_skill_extraction_tool_schema(self, headers: List[str]) -> Dict[str, Any]:
        """
        Returns a Claude-compatible tool schema for extracting matched and missing skills per header.
        """
        base_schema = pydantic_model_to_schema(HeaderSkillsComparison)

        input_schema = {
            "type": "object",
            "description": "Map of each section header to its matched and missing skills.",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }

        for header in headers:
            self.map_header[sanitize_header(header)] = header
            input_schema["properties"][sanitize_header(header)] = base_schema
            input_schema["required"].append(sanitize_header(header))

        return [{
            "name": "match_skills_by_header",
            "description": "Extracts matched and missing skills per section header based on the job description and candidate skill list.",
            "input_schema": input_schema
        }]
        
    
    def extract_skills_grouped_by_header(self, job_description: str, candidate_skills_text: str) -> Dict[str, Dict[str, List[str]]]:
        """
        Extracts and matches skills per header using a single LLM call.
        """
        tools = self.get_skill_extraction_tool_schema(self.headers)
        self.logger.info("Extracting skills grouped by headers with matching and missing info.")

        prompt_content = self._build_prompt(job_description, candidate_skills_text)
        messages = [{
            "role": "user",
            "content": prompt_content
        }]
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                messages=messages,
                tools=tools,
                tool_choice={"type": "tool", "name": "match_skills_by_header"},
            )

            if (
                response.content
                and hasattr(response.content[0], "type")
                and response.content[0].type == "tool_use"
            ):
                tool_input = response.content[0].input
                remapped_input = {
                    self.map_header.get(key, key): val
                    for key, val in tool_input.items()
                }
                self.logger.info("Tool input received from LLM.")
                return remapped_input

            self.logger.error("Tool use response not found.")
            return {header: {"matched_skills": [], "missing_skills": []} for header in self.headers}

        except Exception as e:
            self.logger.error(f"Tool-based skill extraction failed: {e}", exc_info=True)
            return {header: {"matched_skills": [], "missing_skills": []} for header in self.headers}

    def _build_prompt(self, job_description: str, candidate_skills_text: str) -> str:
        return f"""
You are an AI assistant that compares skills required in a job description to the candidate's skill list.

Your goal is to evaluate the candidate's fit under each given headers.

For each header, extract:
- matched_skills: skills from the job posting that match candidate's for this header. You can infer some skills and also if it is under the same technology, you can include it under the matching skills.
- missing_skills: important job skills for this header that are missing in the candidate’s skills.

Job Description:
{job_description}

Candidate's Skillset:
{candidate_skills_text}"""