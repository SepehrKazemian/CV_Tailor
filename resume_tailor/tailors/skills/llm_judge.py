from pydantic import BaseModel, Field
from typing import List
from typing import Dict, Any, List
from resume_tailor.utils.pydantic_to_schema import pydantic_model_to_schema

class JudgeEvaluation(BaseModel):
    """Schema for judging skills section quality."""
    score: int = Field(..., ge=0, le=100, description="Evaluation score from 0 to 100. 90+ means acceptable.")
    comment: str = Field(..., description="Feedback on how to improve the skills section.")

def get_judge_tool_schema() -> List[Dict[str, Any]]:
    """Returns the schema for the Claude-compatible skill section evaluation tool."""
    return [{
        "name": "verify_skills_section",
        "description": "Evaluates the selected skills section against the job description.",
        "parameters": pydantic_model_to_schema(JudgeEvaluation)
    }]