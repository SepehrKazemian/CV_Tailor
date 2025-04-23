from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class HeaderSkillsComparison(BaseModel):
    matched_skills: List[str] = Field(
        ...,
        description="Skills under this header that match both the job and the candidate's skills list.",
    )
    missing_skills: List[str] = Field(
        ...,
        description="Relevant job skills under this header not found in the candidate's skills list.",
    )


class SkillJudgeEvaluation(BaseModel):
    score: int = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Overall numeric score (0–100) reflecting how well the selected skills section "
            "covers the job description and uses appropriate headers."
        ),
    )
    comment: str = Field(
        ...,
        description=(
            "Concise qualitative feedback on the structure, relevance, and completeness "
            "of the selected skills section."
        ),
    )
    suggested_headers_skills: Dict[str, List[str]] = Field(
        ...,
        description=(
            "Suggested replacement or improvement for the skills section. Dictionary containing up to "
            "5 headers (including 'Soft Skills') and the skills that should be added under each."
        ),
    )


class SkillExtractionOutput(BaseModel):
    skills: List[str] = Field(
        ...,
        description="List of job-relevant skills, tools, platforms, or any ATS-based important keywords directly extracted from the job description.",
    )


class SkillMapContent(BaseModel):
    skills: List[str] = Field(
        ...,
        description="List of candidate skills grouped under this specific skill category.",
    )


class SkillMapOutput(BaseModel):
    headers: SkillMapContent = Field(
        ...,
        description="An object containing a single skill category (header) and its related skills. The field name 'headers' is fixed and replaces dynamic headers.",
    )

    
class PrimarySkillMap(BaseModel):
    Primary: Optional[List[str]] = Field(
        default=None,
        description=(
            "A list of technical skills that are clearly and explicitly mentioned anywhere in the job description, regardless of the section. "
            "These are important, must-have, or preferred skills directly stated by the employer. "
            "Return this field as a JSON array of individual skill names (e.g., [\"Python\", \"Docker\"])."
        )    
    )

class InferredSkillMap(BaseModel):
    Inferred: Optional[List[str]] = Field(
        default=None,
        description=(
            "A list of technical skills that are not directly mentioned in the job description, but are strongly implied based on the tools, technologies, or responsibilities discussed. "
            "These skills are commonly used in roles like this, and complement the explicitly listed skills. "
            "Do not include any skills already mentioned. Return this field as a JSON array of individual skill names."
        )
    )

class InferredSecondarySkillMap(BaseModel):
    Inferred_Secondary: Optional[List[str]] = Field(
        default=None,
        description=(
            "A list of less common, niche, or specialized technical skills that are not mentioned in the job description, but may still be contextually useful. "
            "These are weakly inferred and should only include skills relevant to the general role or industry. "
            "Return this field as a JSON array of individual skill names."
        )
    )