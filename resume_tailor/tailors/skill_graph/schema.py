from pydantic import BaseModel, Field
from typing import List, Optional # Import Optional

class ExtractedSkills(BaseModel):
    """Pydantic model for skills extracted by the LLM."""
    skills: List[str] = Field(
        ...,
        description="A list of specific technical skills, tools, libraries, or concepts mentioned in the text."
    )

class L1L2ValidationResult(BaseModel):
    """Pydantic model for the judge LLM's validation of L1 -> L2 connections."""
    selected_l2_categories: List[str] = Field(
        default_factory=list, # Default to empty list if none selected
        description="A list of appropriate L2 category names for the given L1 skill from the provided candidates."
    )
    suggested_l2_category: Optional[str] = Field(
        None,
        description="An alternative, existing L2 category suggested by the judge if none of the candidates were suitable."
    )
    reasoning: Optional[str] = Field(
        None,
        description="Brief explanation for the selections or suggestion." # Optional field for debugging/logging
    )

class L3DeterminationResult(BaseModel):
    """Pydantic model for determining the best L3 domain for an L1 skill."""
    l3_domain: Optional[str] = Field(
        None,
        description="The name of the most appropriate existing L3 domain for the given L1 skill."
    )
    reasoning: Optional[str] = Field(None, description="Explanation for the choice.")

class L2NameDeterminationResult(BaseModel):
    """Pydantic model for determining an L2 category name."""
    l2_category_name: Optional[str] = Field(
        None,
        description="A suitable L2 category name (e.g., 'Frameworks', 'Libraries', 'Platforms', 'Tools', 'Concepts') based on the L1 skill and its L3 domain."
    )
    reasoning: Optional[str] = Field(None, description="Explanation for the choice.")

class GeneratedL2Categories(BaseModel):
    """Pydantic model for L2 categories generated for a new L3 domain."""
    l2_category_names: List[str] = Field(
        default_factory=list,
        description="A list of standard L2 category names (e.g., Tools, Frameworks, Platforms, Concepts) relevant to the given L3 domain."
    )


# Example Usage:
# try:
#     data = {"skills": ["Python", "PyTorch", "AWS S3", "Langchain", "Docker"]}
#     extracted = ExtractedSkills(**data)
#     print(extracted.skills)
# except ValidationError as e:
#     print(e)
