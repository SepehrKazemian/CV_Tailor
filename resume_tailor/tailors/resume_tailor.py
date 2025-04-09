"""
Resume Tailor Module - Orchestrator

This module orchestrates the resume tailoring process by coordinating
different tailor components (summary, skills, experience).
"""

from typing import Optional

# Import utility functions moved to tailor_utils
from resume_tailor.tailors.tailor_utils import (
    ensure_page_limit  # Removed find_missing_keywords, TECH_SUBS, flatten_values import
)
from resume_tailor.analyzers.job_analyzer import JobAnalyzer
# Import the section-specific tailors
from resume_tailor.tailors.professional_summary.tailor import ProfessionalSummaryTailor
from resume_tailor.tailors.skills_tailor import SkillsRefiner
from resume_tailor.tailors.experience_tailor import ExperienceRefiner # Corrected import name
from resume_tailor.utils.llm_provider import get_llm
from resume_tailor.utils.file_utils import (
    read_resume_components, # This now reads YAML for experience
    read_job_description,
    save_warnings,
    save_substitutions,
)
from resume_tailor.output.docx_generator import generate_docx


class ResumeTailor:
    """
    Orchestrates the tailoring of resume components based on job analysis.
    """

    def __init__(
        self,
        cv_dir: str = "cv",
        llm_provider: str = "openai",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the ResumeTailor orchestrator.

        Args:
            cv_dir (str): Directory containing resume component files.
            llm_provider (str): LLM provider to use.
            api_key (Optional[str]): API key for the selected LLM provider.
        """
        self.cv_dir = cv_dir
        self.llm_provider = llm_provider
        self.api_key = api_key
        # Get LLM instances needed by components
        _, self.llm, _ = get_llm(llm_provider, api_key, return_raw=True)

        # Instantiate components
        self.job_analyzer = JobAnalyzer(llm_provider, api_key, cv_dir)
        self.professional_summary_tailor = ProfessionalSummaryTailor(llm_provider, api_key)
        self.skills_refiner = SkillsRefiner(llm_provider, api_key)
        self.experience_refiner = ExperienceRefiner(llm_provider, api_key) # Use correct class and pass provider/key

        # Load base resume components (read_resume_components now handles YAML for experience)
        self.resume_components = read_resume_components(cv_dir)
        self.warnings = []
        self.substitutions = {} # Substitutions are no longer handled by skills refiner

    # Removed tailor_professional_summary, tailor_skills, tailor_experience methods
    # Removed ensure_page_limit, flatten_values, find_missing_keywords methods (moved to utils)

    def generate_tailored_resume(self, job_description: str) -> str:
        """
        Generate a tailored resume based on the job description.

        Args:
            job_description: The job description text

        Returns:
            The complete tailored resume as a string
        """
        # 1. Analyze Job Description
        analysis = self.job_analyzer.analyze(job_description)
        if not analysis or not isinstance(analysis, dict):
            self.warnings.append("Job analysis failed. Cannot tailor resume.")
            return "\n\n".join(self.resume_components.values())

        # 2. Refine Professional Summary
        original_summary = self.resume_components.get("professional_summary", "")
        tailored_summary = self.professional_summary_tailor.refine_summary(
            analysis, original_summary, job_description
        )
        # TODO: Collect warnings from sub-tailors if they implement a warnings list
        # self.warnings.extend(self.professional_summary_tailor.warnings)

        # 3. Refine Skills Section
        original_skills = self.resume_components.get("skills", "")
        # The new refiner handles finding missing keywords internally if needed by its prompts
        # It returns the final formatted string directly. Substitutions are no longer handled here.
        tailored_skills = self.skills_refiner.refine_skills(
            job_description, original_skills  # Pass job description and original skills
        )
        # self.warnings.extend(self.skills_refiner.warnings) # Collect warnings if implemented

        # 4. Refine Experience Section
        original_experience_data = self.resume_components.get("experience", {})  # Expecting dict from YAML
        # Pass the parsed dictionary, job description, and analysis to the refiner
        tailored_experience = self.experience_refiner.refine_experience(
            original_experience_data, job_description, analysis
        )
        # self.warnings.extend(self.experience_refiner.warnings)  # Collect warnings if implemented

        # 5. Get Education (typically unchanged)
        education = self.resume_components.get("education", "")

        # 6. Combine Sections
        resume = f"""# PROFESSIONAL SUMMARY
{tailored_summary}

# SKILLS
{tailored_skills}

# WORK EXPERIENCE
{tailored_experience}

# EDUCATION
{education}
"""

        # 7. Ensure Page Limit using utility function
        # Pass self.llm and self.warnings list to the utility function
        resume = ensure_page_limit(resume, self.llm, self.warnings)

        return resume

    def tailor_resume(
        self,
        job_file: Optional[str] = None,
        job_text: Optional[str] = None,
        output_path: str = "tailored_resume.docx",
        format: str = "docx",
    ) -> str:
        """
        Main entry point: Tailor a resume and save it.

        Args:
            job_file: Path to job description file.
            job_text: Job description text.
            output_path: Path to save the tailored resume.
            format: Output format ('docx', 'txt', or 'md').

        Returns:
            The path to the saved resume file, or empty string on failure.
        """
        job_description = read_job_description(job_file, job_text)
        if not job_description:
            print("Error: Could not read job description.")
            return ""

        tailored_resume = self.generate_tailored_resume(job_description)

        # Save the resume
        try:
            if format == "docx":
                generate_docx(tailored_resume, output_path)
            else:
                with open(output_path, "w") as file:
                    file.write(tailored_resume)
            print(f"Tailored resume saved to {output_path}")
        except Exception as e:
            print(f"Error saving resume to {output_path}: {e}")
            self.warnings.append(f"Failed to save resume: {e}")
            return ""

        # Save warnings and substitutions
        # TODO: Consider passing output dir to save_warnings/save_substitutions
        if self.warnings:
            save_warnings(self.warnings)
        if self.substitutions:
            # Note: Substitutions dict might be empty now as it's not generated by SkillsRefiner
            save_substitutions(self.substitutions)

        return output_path

    # Utility methods moved to tailor_utils.py
