from typing import List, Dict
from pydantic import BaseModel, Field

class HeaderSkillsComparison(BaseModel):
    matched_skills: List[str] = Field(
        ..., description="Skills under this header that match both the job and the candidate's skills list."
    )
    missing_skills: List[str] = Field(
        ..., description="Relevant job skills under this header not found in the candidate's skills list."
    )