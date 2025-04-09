"""
ATS Evaluator Module

This module provides functionality to evaluate tailored resumes using an LLM as an ATS system.
It checks if skills from the job description are properly included in the resume and handles
technology substitutions intelligently.
"""

import os
import json
import importlib.util
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Import config.py dynamically
config_path = Path(__file__).resolve().parents[2] / "config.py"
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)


@dataclass
class ResumeEvaluation:
    """Class to store the evaluation results of a resume."""
    
    # Overall evaluation
    ats_score: float = 0.0  # 0.0 to 1.0
    overall_feedback: str = ""
    
    # Skills evaluation
    required_skills: List[str] = field(default_factory=list)
    found_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    
    # Technology substitutions
    substitutions: Dict[str, str] = field(default_factory=dict)  # original -> substitute
    
    # Section-specific feedback
    professional_summary_feedback: str = ""
    skills_feedback: str = ""
    experience_feedback: str = ""
    education_feedback: str = ""
    
    # Improvement suggestions
    improvement_suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the evaluation to a dictionary."""
        return {
            "ats_score": self.ats_score,
            "overall_feedback": self.overall_feedback,
            "required_skills": self.required_skills,
            "found_skills": self.found_skills,
            "missing_skills": self.missing_skills,
            "substitutions": self.substitutions,
            "section_feedback": {
                "professional_summary": self.professional_summary_feedback,
                "skills": self.skills_feedback,
                "experience": self.experience_feedback,
                "education": self.education_feedback
            },
            "improvement_suggestions": self.improvement_suggestions
        }
    
    def save_to_file(self, output_path: str) -> None:
        """Save the evaluation to a JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def print_summary(self) -> None:
        """Print a summary of the evaluation."""
        print(f"ATS Score: {self.ats_score:.2f} (0.0-1.0)")
        print(f"Overall Feedback: {self.overall_feedback}")
        print("\nSkills Analysis:")
        print(f"  Required Skills: {', '.join(self.required_skills)}")
        print(f"  Found Skills: {', '.join(self.found_skills)}")
        print(f"  Missing Skills: {', '.join(self.missing_skills)}")
        
        if self.substitutions:
            print("\nTechnology Substitutions:")
            for original, substitute in self.substitutions.items():
                print(f"  {original} -> {substitute}")
        
        print("\nImprovement Suggestions:")
        for i, suggestion in enumerate(self.improvement_suggestions, 1):
            print(f"  {i}. {suggestion}")


class ATSEvaluationResult(BaseModel):
    """Structured output for ATS evaluation."""
    ats_score: float = Field(description="ATS score from 0.0 to 1.0")
    overall_feedback: str = Field(description="Overall feedback on the resume")
    required_skills: List[str] = Field(description="Skills required by the job description")
    found_skills: List[str] = Field(description="Skills found in the resume")
    missing_skills: List[str] = Field(description="Skills missing from the resume")
    substitutions: Dict[str, str] = Field(description="Technology substitutions found (original -> substitute)")
    professional_summary_feedback: str = Field(description="Feedback on the professional summary section")
    skills_feedback: str = Field(description="Feedback on the skills section")
    experience_feedback: str = Field(description="Feedback on the experience section")
    education_feedback: str = Field(description="Feedback on the education section")
    improvement_suggestions: List[str] = Field(description="Suggestions for improving the resume")


class ATSEvaluator:
    """
    Evaluates tailored resumes using an LLM as an ATS system.
    """
    
    def __init__(self, llm_provider: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the ATSEvaluator.
        
        Args:
            llm_provider: LLM provider to use ('openai', 'google', 'anthropic', or None to use config.py)
            api_key: API key for the selected LLM provider (None to use config.py or environment variable)
        """
        self.llm_provider = llm_provider or getattr(config, 'LLM', 'claude').lower()
        self.api_key = api_key or getattr(config, 'key', None)
        self.llm_version = getattr(config, 'version', None)
        
        # Initialize the LLM
        self.llm = self._initialize_llm()
        
        # Technology substitution mapping (can be expanded)
        self.tech_substitutions = {
            "aws": ["gcp", "azure", "cloud"],
            "gcp": ["aws", "azure", "cloud"],
            "azure": ["aws", "gcp", "cloud"],
            "python": ["java", "c++", "golang", "javascript"],
            "tensorflow": ["pytorch", "keras", "mxnet"],
            "pytorch": ["tensorflow", "keras", "mxnet"],
            "react": ["angular", "vue", "svelte"],
            "javascript": ["typescript", "python", "java"],
            "docker": ["kubernetes", "containerization"],
            "kubernetes": ["docker", "containerization"],
            "sql": ["postgresql", "mysql", "oracle", "nosql"],
            "nosql": ["mongodb", "dynamodb", "cassandra"],
            "fastapi": ["flask", "django", "express"],
            "langchain": ["llamaindex", "semantic-kernel"],
            "llamaindex": ["langchain", "semantic-kernel"],
            "openai": ["anthropic", "gemini", "llama", "mistral"],
            "gemini": ["openai", "anthropic", "llama", "mistral"],
            "anthropic": ["openai", "gemini", "llama", "mistral"],
            "mlflow": ["wandb", "tensorboard", "neptune"],
            "airflow": ["dagster", "prefect", "luigi"],
            "dagster": ["airflow", "prefect", "luigi"],
            "snowflake": ["bigquery", "redshift", "databricks"],
            "bigquery": ["snowflake", "redshift", "databricks"],
            "databricks": ["snowflake", "bigquery", "redshift"],
            "streamlit": ["dash", "gradio", "panel"],
            "tableau": ["power bi", "looker", "quicksight"],
            "power bi": ["tableau", "looker", "quicksight"],
            "elasticsearch": ["opensearch", "solr", "algolia"],
            "qdrant": ["pinecone", "weaviate", "milvus", "vector database"],
            "pinecone": ["qdrant", "weaviate", "milvus", "vector database"],
            "weaviate": ["qdrant", "pinecone", "milvus", "vector database"],
        }
    
    def _initialize_llm(self):
        """Initialize the LLM based on the provider."""
        if self.llm_provider == "openai":
            return ChatOpenAI(
                model="gpt-4" if not self.llm_version else f"gpt-{self.llm_version}",
                temperature=0.2,
                api_key=self.api_key or os.environ.get("OPENAI_API_KEY")
            )
        elif self.llm_provider == "google":
            return ChatGoogleGenerativeAI(
                model="gemini-pro" if not self.llm_version else f"gemini-{self.llm_version}",
                temperature=0.2,
                api_key=self.api_key or os.environ.get("GOOGLE_API_KEY")
            )
        elif self.llm_provider == "claude" or self.llm_provider == "anthropic":
            # Use a standard model name that's known to work
            model_version = "claude-3-opus-20240229"
            
            return ChatAnthropic(
                model=model_version,
                temperature=0.2,
                api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY")
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def evaluate_resume(self, job_description: str, resume_text: str, 
                        original_resume_components: Dict[str, str]) -> ResumeEvaluation:
        """
        Evaluate a tailored resume against a job description.
        
        Args:
            job_description: The job description text
            resume_text: The tailored resume text
            original_resume_components: The original resume components
            
        Returns:
            A ResumeEvaluation object containing the evaluation results
        """
        # Create a parser for structured output
        parser = PydanticOutputParser(pydantic_object=ATSEvaluationResult)
        
        # Create a prompt template
        prompt_template = PromptTemplate(
            template="""
            You are an expert ATS (Applicant Tracking System) evaluator with deep knowledge of hiring practices.
            
            I need you to evaluate a resume against a job description and provide detailed feedback.
            
            # Job Description:
            {job_description}
            
            # Tailored Resume:
            {resume_text}
            
            # Original Resume Components:
            ## Professional Summary:
            {original_summary}
            
            ## Skills:
            {original_skills}
            
            ## Experience:
            {original_experience}
            
            ## Education:
            {original_education}
            
            Evaluate the tailored resume against the job description, considering:
            1. How well the resume matches the job requirements
            2. If required skills are present in the resume
            3. If technology substitutions are appropriate (e.g., if the job requires AWS but the resume mentions GCP experience)
            4. The overall quality and effectiveness of the resume
            
            For technology substitutions, consider these as equivalent:
            {tech_substitutions}
            
            Important: Only consider skills as "found" if they are actually in the resume OR if there is an appropriate substitution.
            If a skill is required but has no equivalent in the original resume components, it should be marked as "missing".
            
            {format_instructions}
            """,
            input_variables=["job_description", "resume_text", "original_summary", 
                            "original_skills", "original_experience", "original_education",
                            "tech_substitutions"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        # Create the chain
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        
        # Format technology substitutions for the prompt
        tech_subs_formatted = []
        for tech, alternatives in self.tech_substitutions.items():
            tech_subs_formatted.append(f"- {tech}: {', '.join(alternatives)}")
        
        try:
            # Run the chain
            result = chain.run(
                job_description=job_description,
                resume_text=resume_text,
                original_summary=original_resume_components.get("professional_summary", ""),
                original_skills=original_resume_components.get("skills", ""),
                original_experience=original_resume_components.get("experience", ""),
                original_education=original_resume_components.get("education", ""),
                tech_substitutions="\n".join(tech_subs_formatted)
            )
            
            # Try to fix the JSON by adding missing commas
            try:
                # Parse the result
                parsed_result = parser.parse(result)
                
                # Convert to ResumeEvaluation
                evaluation = ResumeEvaluation(
                    ats_score=parsed_result.ats_score,
                    overall_feedback=parsed_result.overall_feedback,
                    required_skills=parsed_result.required_skills,
                    found_skills=parsed_result.found_skills,
                    missing_skills=parsed_result.missing_skills,
                    substitutions=parsed_result.substitutions,
                    professional_summary_feedback=parsed_result.professional_summary_feedback,
                    skills_feedback=parsed_result.skills_feedback,
                    experience_feedback=parsed_result.experience_feedback,
                    education_feedback=parsed_result.education_feedback,
                    improvement_suggestions=parsed_result.improvement_suggestions
                )
            except Exception as json_error:
                print(f"Error parsing JSON: {json_error}")
                print("Attempting to fix JSON format...")
                
                # Try to extract JSON from the response
                import re
                import json
                
                # Find JSON-like content between curly braces
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    
                    # Add missing commas between array elements
                    json_str = re.sub(r'"\s+(")', '", \1', json_str)
                    
                    # Add missing commas between key-value pairs
                    json_str = re.sub(r'"\s+}', '"}', json_str)
                    json_str = re.sub(r'"\s+"', '", "', json_str)
                    json_str = re.sub(r'}\s+"', '}, "', json_str)
                    json_str = re.sub(r']\s+"', '], "', json_str)
                    
                    try:
                        # Parse the fixed JSON
                        fixed_data = json.loads(json_str)
                        
                        # Convert to ResumeEvaluation
                        evaluation = ResumeEvaluation(
                            ats_score=fixed_data.get("ats_score", 0.0),
                            overall_feedback=fixed_data.get("overall_feedback", ""),
                            required_skills=fixed_data.get("required_skills", []),
                            found_skills=fixed_data.get("found_skills", []),
                            missing_skills=fixed_data.get("missing_skills", []),
                            substitutions=fixed_data.get("substitutions", {}),
                            professional_summary_feedback=fixed_data.get("professional_summary_feedback", ""),
                            skills_feedback=fixed_data.get("skills_feedback", ""),
                            experience_feedback=fixed_data.get("experience_feedback", ""),
                            education_feedback=fixed_data.get("education_feedback", ""),
                            improvement_suggestions=fixed_data.get("improvement_suggestions", [])
                        )
                    except json.JSONDecodeError as e:
                        print(f"Failed to fix JSON: {e}")
                        raise
            
            return evaluation
            
        except Exception as e:
            print(f"Error evaluating resume: {e}")
            # Return a basic evaluation if the chain fails
            return ResumeEvaluation(
                ats_score=0.0,
                overall_feedback=f"Error evaluating resume: {e}",
                improvement_suggestions=["Fix the evaluation error and try again."]
            )
    
    def get_improvement_feedback(self, evaluation: ResumeEvaluation, 
                                job_description: str, resume_text: str) -> str:
        """
        Get detailed feedback on how to improve the resume.
        
        Args:
            evaluation: The resume evaluation
            job_description: The job description text
            resume_text: The tailored resume text
            
        Returns:
            Detailed feedback on how to improve the resume
        """
        prompt_template = PromptTemplate(
            template="""
            You are an expert resume writer with deep knowledge of ATS systems and hiring practices.
            
            Based on the evaluation of a resume against a job description, provide detailed feedback on how to improve the resume.
            
            # Job Description:
            {job_description}
            
            # Tailored Resume:
            {resume_text}
            
            # Evaluation:
            - ATS Score: {ats_score}
            - Overall Feedback: {overall_feedback}
            - Missing Skills: {missing_skills}
            - Professional Summary Feedback: {professional_summary_feedback}
            - Skills Feedback: {skills_feedback}
            - Experience Feedback: {experience_feedback}
            - Education Feedback: {education_feedback}
            
            Provide detailed, actionable feedback on how to improve the resume to better match the job description.
            Focus on:
            1. How to address the missing skills
            2. How to improve each section
            3. Specific wording or formatting changes
            4. Any other improvements that would increase the ATS score
            
            Your feedback should be specific, actionable, and tailored to this particular resume and job description.
            """,
            input_variables=["job_description", "resume_text", "ats_score", "overall_feedback",
                            "missing_skills", "professional_summary_feedback", "skills_feedback",
                            "experience_feedback", "education_feedback"]
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        
        try:
            result = chain.run(
                job_description=job_description,
                resume_text=resume_text,
                ats_score=evaluation.ats_score,
                overall_feedback=evaluation.overall_feedback,
                missing_skills=", ".join(evaluation.missing_skills),
                professional_summary_feedback=evaluation.professional_summary_feedback,
                skills_feedback=evaluation.skills_feedback,
                experience_feedback=evaluation.experience_feedback,
                education_feedback=evaluation.education_feedback
            )
            
            return result.strip()
            
        except Exception as e:
            print(f"Error getting improvement feedback: {e}")
            return f"Error getting improvement feedback: {e}"
    
    def is_skill_in_resume(self, skill: str, resume_text: str, 
                          original_resume_components: Dict[str, str]) -> Tuple[bool, Optional[str]]:
        """
        Check if a skill is in the resume or if there is an appropriate substitution.
        
        Args:
            skill: The skill to check for
            resume_text: The tailored resume text
            original_resume_components: The original resume components
            
        Returns:
            A tuple of (is_found, substitution) where:
            - is_found: True if the skill is found in the resume
            - substitution: The substitution used, if any
        """
        # Check if the skill is directly in the resume
        if skill.lower() in resume_text.lower():
            return True, None
        
        # Check for substitutions
        skill_lower = skill.lower()
        
        # Check if the skill is in the tech substitutions
        if skill_lower in self.tech_substitutions:
            # Check if any of the alternatives are in the resume
            for alt in self.tech_substitutions[skill_lower]:
                if alt.lower() in resume_text.lower():
                    return True, alt
        
        # Check if the skill is an alternative for any tech in the original resume
        for tech, alternatives in self.tech_substitutions.items():
            if skill_lower in [alt.lower() for alt in alternatives]:
                # Check if the original tech is in the resume
                if tech.lower() in resume_text.lower():
                    return True, tech
                
                # Check if the original tech is in the original resume components
                original_text = " ".join(original_resume_components.values()).lower()
                if tech.lower() in original_text:
                    # Check if any other alternative is in the resume
                    for alt in alternatives:
                        if alt.lower() != skill_lower and alt.lower() in resume_text.lower():
                            return True, alt
        
        return False, None
    
    def extract_skills_from_job_description(self, job_description: str) -> Set[str]:
        """
        Extract skills from a job description.
        
        Args:
            job_description: The job description text
            
        Returns:
            A set of skills extracted from the job description
        """
        prompt_template = PromptTemplate(
            template="""
            You are an expert at extracting technical skills and requirements from job descriptions.
            
            Extract all the technical skills, tools, technologies, and qualifications from the following job description.
            Return them as a comma-separated list of individual skills (e.g., "Python, JavaScript, AWS, Docker").
            
            Job Description:
            {job_description}
            
            Skills:
            """,
            input_variables=["job_description"]
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        
        try:
            result = chain.run(job_description=job_description)
            
            # Parse the result into a set of skills
            skills = {skill.strip().lower() for skill in result.split(",") if skill.strip()}
            return skills
            
        except Exception as e:
            print(f"Error extracting skills from job description: {e}")
            return set()
