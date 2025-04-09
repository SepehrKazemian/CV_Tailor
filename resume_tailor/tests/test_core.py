"""
Core Tests for Resume Tailor

This module contains tests for the core functionality of the resume tailoring pipeline.
"""

import os
import pytest
from resume_tailor.analyzers.job_analyzer import JobAnalyzer
from resume_tailor.tailors.resume_tailor import ResumeTailor
from resume_tailor.output.docx_generator import estimate_pages
from resume_tailor.utils.file_utils import read_job_description


def test_job_analyzer():
    """Test the JobAnalyzer class."""
    # Skip if no API key is set
    if not any(key in os.environ for key in ["OPENAI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"]):
        pytest.skip("No API key set for any LLM provider")
    
    # Use the first available API key
    if "OPENAI_API_KEY" in os.environ:
        llm_provider = "openai"
    elif "GOOGLE_API_KEY" in os.environ:
        llm_provider = "google"
    elif "ANTHROPIC_API_KEY" in os.environ:
        llm_provider = "anthropic"
    
    # Initialize the job analyzer
    analyzer = JobAnalyzer(llm_provider=llm_provider)
    
    # Read the example job description
    job_description = read_job_description("example_job.txt")
    
    # Analyze the job description
    analysis = analyzer.analyze(job_description)
    
    # Check that the analysis contains the expected keys
    expected_keys = [
        "keywords",
        "soft_skills",
        "domain_knowledge",
        "experience_focus",
        "tech_stack",
        "related_technologies",
        "priority_order",
        "tone"
    ]
    
    for key in expected_keys:
        assert key in analysis, f"Analysis missing key: {key}"
    
    # Check that the analysis contains non-empty values
    for key in expected_keys:
        if key != "tone":  # tone can be a string
            assert isinstance(analysis[key], list), f"Analysis key {key} is not a list"
            assert len(analysis[key]) > 0, f"Analysis key {key} is empty"
        else:
            assert isinstance(analysis[key], str), f"Analysis key {key} is not a string"
            assert analysis[key], f"Analysis key {key} is empty"


def test_resume_tailor():
    """Test the ResumeTailor class."""
    # Skip if no API key is set
    if not any(key in os.environ for key in ["OPENAI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"]):
        pytest.skip("No API key set for any LLM provider")
    
    # Use the first available API key
    if "OPENAI_API_KEY" in os.environ:
        llm_provider = "openai"
    elif "GOOGLE_API_KEY" in os.environ:
        llm_provider = "google"
    elif "ANTHROPIC_API_KEY" in os.environ:
        llm_provider = "anthropic"
    
    # Initialize the resume tailor
    tailor = ResumeTailor(cv_dir="cv", llm_provider=llm_provider)
    
    # Read the example job description
    job_description = read_job_description("example_job.txt")
    
    # Generate a tailored resume
    tailored_resume = tailor.generate_tailored_resume(job_description)
    
    # Check that the tailored resume contains all required sections
    required_sections = ["# PROFESSIONAL SUMMARY", "# SKILLS", "# WORK EXPERIENCE", "# EDUCATION"]
    for section in required_sections:
        assert section in tailored_resume, f"Tailored resume missing section: {section}"
    
    # Check that the tailored resume is not too long
    pages = estimate_pages(tailored_resume)
    assert pages <= 2, f"Tailored resume is too long: {pages} pages"


def test_docx_generator():
    """Test the docx_generator module."""
    from resume_tailor.output.docx_generator import generate_docx
    
    # Create a simple resume
    resume = """# PROFESSIONAL SUMMARY
A professional summary.

# SKILLS
• Skill 1
• Skill 2

# WORK EXPERIENCE
Work experience.

# EDUCATION
Education.
"""
    
    # Generate a docx file
    output_path = "test_docx.docx"
    generate_docx(resume, output_path)
    
    # Check that the file was created
    assert os.path.exists(output_path), f"DOCX file not created: {output_path}"
    
    # Clean up
    os.remove(output_path)


def test_file_utils():
    """Test the file_utils module."""
    from resume_tailor.utils.file_utils import read_resume_components
    
    # Read the resume components
    components = read_resume_components("cv")
    
    # Check that all components were read
    expected_components = ["professional_summary", "skills", "experience", "education"]
    for component in expected_components:
        assert component in components, f"Components missing key: {component}"
        assert components[component], f"Component {component} is empty"
