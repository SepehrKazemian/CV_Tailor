"""
Job Analyzer Module

This module handles the analysis of job descriptions using LLMs to extract
keywords, skills, and other important information.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, get_origin, get_args
from enum import Enum

from pydantic import BaseModel, Field
from resume_tailor.utils.llm_provider import get_llm

# Set up logging
logger = logging.getLogger(__name__)
# Configure root logger if not already configured
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO)


class ToneEnum(str, Enum):
    technical = "technical"
    business_focused = "business-focused"
    innovative = "innovative"


class JobAnalysis(BaseModel):
    """Analysis of a job description for resume tailoring."""

    keywords: List[str] = Field(
        description="A list of most important technical skills, tools, and technologies mentioned for the job posting "
                    "and the company"
    )
    soft_skills: List[str] = Field(
        description="A list of soft skills and qualities emphasized"
    )
    domain_knowledge: List[str] = Field(
        description="A list of specific industry or domain knowledge required"
    )
    experience_focus: List[str] = Field(
        description="A list of aspects of work experience that should be emphasized"
    )
    tech_stack: List[str] = Field(
        description="A list of primary technologies that form the core tech stack"
    )
    related_technologies: List[str] = Field(
        description="A list of technologies not explicitly mentioned but likely relevant"
    )
    priority_order: List[str] = Field(
        description="Suggested order of importance for skills to highlight (most important first)."
    )
    responsibilities: List[str] = Field(
        description="A list of responsibilities for the ideal candidate of this role based on Job Description."
    )
    business: str = Field(
        description="The industry or business domain of the company, inferred from the job description and company name."
    )
    tone: ToneEnum = Field(description="Suggested tone for the resume")


class JobAnalyzer:
    """
    Analyzes job descriptions to extract keywords and other important information.
    """

    def __init__(self, llm_provider: str = "openai", api_key: Optional[str] = None, cv_dir: str = ""):
        """
        Initialize the JobAnalyzer.

        Args:
            llm_provider: LLM provider to use ('openai', 'google', 'anthropic', 'claude').
            api_key: API key for the selected LLM provider.
            cv_dir: Directory containing CV component files for keyword extraction.
        """
        # Get all three return values, but JobAnalyzer primarily uses model_name and raw_llm_client
        self.model_name, _, self.llm = get_llm(llm_provider, api_key, return_raw=True)
        if not self.llm:
            raise ValueError(
                f"Could not initialize raw LLM client for provider {llm_provider}. Job analysis requires tool calling."
            )
        self.cv_dir = cv_dir
        logger.info(f"Initialized JobAnalyzer with model: {self.model_name}")
        # Load CV keywords during initialization
        self.experience_keywords = self.load_or_extract_cv_keywords("experience.txt")
        self.skills_keywords = self.load_or_extract_cv_keywords("skills.txt")
        self.summary_keywords = self.load_or_extract_cv_keywords("ps.txt")

    def tools_calling(self):
        """
        Build a Claude-compatible tool schema from the JobAnalysis model.

        Returns:
            List[Dict[str, Any]]: Claude tool definition with input schema.
        """
        properties = {}
        required_fields = []

        for field_name, model_field in JobAnalysis.model_fields.items():
            required_fields.append(field_name)
            field_type = model_field.annotation

            actual_type = field_type
            origin = get_origin(actual_type)

            if isinstance(actual_type, type) and issubclass(actual_type, Enum):
                # Single enum field
                schema_type = {
                    "type": "string",
                    "description": model_field.description or "",
                    "enum": [e.value for e in actual_type],
                }

            elif origin in (list, List):
                item_type = get_args(actual_type)[0]

                if isinstance(item_type, type) and issubclass(item_type, Enum):
                    # List of Enums
                    schema_type = {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [e.value for e in item_type],
                        },
                        "description": model_field.description or "",
                    }
                else:
                    # List of strings
                    schema_type = {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": model_field.description or "",
                    }

            else:
                # Default string field
                schema_type = {
                    "type": "string",
                    "description": model_field.description or "",
                }

            properties[field_name] = schema_type

        schema = {
            "type": "object",
            "properties": properties,
            "required": required_fields,
        }

        return [{
            "name": "extract_job_info",
            "description": "Extracts structured information from a job description.",
            "input_schema": schema,
        }]

    def prompt(self, job_description):
        messages = [
            {
                "role": "user",
                "content": (
                    "Analyze the following job description and extract structured information:\n\n"
                    f"{job_description}. The value for each key within the JSON must be an array, "
                    "except for 'business' and 'tone'."
                ),
            }
        ]
        return messages

    def analyze(self, job_description: str) -> Dict[str, Any]:
        """
        Analyze a job description to extract keywords and other important information.

        Args:
            job_description: The job description text

        Returns:
            A dictionary containing the analysis results
        """
        tools = self.tools_calling()
        messages = self.prompt(job_description)

        try:
            response = self.llm.messages.create(
                model=self.model_name,
                max_tokens=1000,
                messages=messages,
                tools=tools,
                tool_choice={"type": "tool", "name": "extract_job_info"},
            )
            # Check if response content is valid and has the expected structure
            if response.content and hasattr(response.content[0], 'input'):
                return response.content[0].input
            else:
                logger.error(f"Unexpected response structure from LLM during analysis: {response.content}")
                return {}  # Return empty dict on unexpected structure

        except Exception as e:
            logger.error(f"Error during job analysis LLM call: {e}", exc_info=True)
            return {}  # Return empty dict on error

    def load_or_extract_cv_keywords(self, file_basename: str, rewrite: bool = False) -> dict:
        """
        Load keywords from a cached *_keywords.txt file if it exists.
        If not, for skills.txt extract structured keywords from text lines.
        For other files, call self.analyze() to get keyword dict.

        Args:
            file_basename (str): Basename of the input file (e.g. 'skills.txt')
            rewrite (bool): If True, regenerate keywords even if cache exists

        Returns:
            dict: Extracted or loaded keyword dictionary
        """
        if not self.cv_dir:
            logger.error("CV directory not set in JobAnalyzer.")
            return {}

        full_file_path = os.path.join(self.cv_dir, file_basename)
        base, _ = os.path.splitext(full_file_path)
        keyword_file = f"{base}_keywords.txt"

        # Use cached version if available and rewrite=False
        if os.path.exists(keyword_file) and not rewrite:
            try:
                with open(keyword_file, "r", encoding="utf-8") as f:
                    logger.info(f"Loading cached keywords from {keyword_file}")
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not load cached keyword file {keyword_file}: {e}. Regenerating.")

        # Load raw text
        try:
            with open(full_file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except FileNotFoundError:
            logger.error(f"CV component file not found: {full_file_path}")
            return {}
        except OSError as e:
            logger.error(f"Error reading CV component file {full_file_path}: {e}")
            return {}

        keywords = {}
        if file_basename == "skills.txt":
            # Custom parsing for skills.txt
            logger.info(f"Extracting keywords via custom parsing for {file_basename}")
            keyword_set = set()
            for line in text.splitlines():
                if ":" in line:
                    _, items = line.split(":", 1)
                    split_items = [item.strip() for item in items.split(",") if item.strip()]
                    keyword_set.update(split_items)
            keywords = {"keywords": sorted(keyword_set)}  # Structure as dict

        elif file_basename in ["ps.txt", "experience.txt"]:
            # Use LLM analysis for summary and experience
            logger.info(f"Extracting keywords via LLM analysis for {file_basename}")
            # Placeholder: Implement a simpler keyword extraction or use basic text processing.
            logger.warning(f"LLM-based keyword extraction for {file_basename} not implemented yet. Returning empty.")
            keywords = {"keywords": []}  # Placeholder structure
            # keywords = self.analyze(text) # This would call the full job analysis on CV parts

        # Save to cache
        try:
            with open(keyword_file, "w", encoding="utf-8") as f:
                json.dump(keywords, f, indent=2)
            logger.info(f"Saved extracted keywords to cache: {keyword_file}")
        except OSError as e:
            logger.error(f"Could not save keyword cache file {keyword_file}: {e}")

        return keywords
