from pydantic import BaseModel, Field

class LengthCheck(BaseModel):
    pass_field: bool = Field(
        ..., 
        alias="pass", 
        description="Pass if the summary is under ~100 words or 5 lines, and feels concise and readable."
    )
    comment: str = Field(
        ..., 
        description="Suggest trimming only if noticeably too long. Do not nitpick borderline cases."
    )
class ContentAlignment(BaseModel):
    score: int = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Score from 0 to 100. 90+ means highly aligned. Only deduct points for meaningful content gaps, not optional additions."
    )
    comment: str = Field(
        ..., 
        description="Suggest only essential additions. If length is tight, suggest what to replace instead of just adding."
    )

class BestPracticesCheck(BaseModel):
    score: int = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Score from 0 to 100. 90+ means strong and polished. Only deduct if there's a real clarity, tone, or structure issue — not minor preferences."
    )
    comment: str = Field(..., description="Suggestions to improve tone, clarity, and specificity. Avoid numbers or excessive detail.")

class ProfessionalSummaryEvaluation(BaseModel):
    length_check: LengthCheck
    content_alignment: ContentAlignment
    best_practices_check: BestPracticesCheck

