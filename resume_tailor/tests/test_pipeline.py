"""
Test Pipeline Module

This module tests the resume tailoring pipeline to ensure it meets all requirements.
"""

import os
import sys
import json
from typing import Dict, List, Any, Optional

from resume_tailor.tailors.resume_tailor import ResumeTailor
from resume_tailor.analyzers.job_analyzer import JobAnalyzer
from resume_tailor.output.docx_generator import estimate_pages
from resume_tailor.utils.file_utils import read_job_description


def test_resume_tailoring(job_file: str = "example_job.txt", 
                         output_path: str = "test_resume.docx",
                         cv_dir: str = "cv",
                         llm_provider: str = "openai") -> Dict[str, Any]:
    """
    Test the resume tailoring pipeline.
    
    Args:
        job_file: Path to the job description file
        output_path: Path to save the tailored resume to
        cv_dir: Directory containing resume component files
        llm_provider: LLM provider to use
        
    Returns:
        A dictionary containing test results
    """
    print(f"Testing resume tailoring pipeline with job description from {job_file}...")
    
    # Read the job description
    try:
        job_description = read_job_description(job_file)
    except FileNotFoundError:
        print(f"Error: Job description file {job_file} not found.")
        return {"success": False, "error": f"Job description file {job_file} not found."}
    
    # Initialize the job analyzer
    job_analyzer = JobAnalyzer(llm_provider=llm_provider)
    
    # Analyze the job description
    print("Analyzing job description...")
    analysis = job_analyzer.analyze(job_description)
    
    # Initialize the resume tailor
    tailor = ResumeTailor(cv_dir=cv_dir, llm_provider=llm_provider)
    
    # Generate the tailored resume
    print("Generating tailored resume...")
    tailored_resume = tailor.generate_tailored_resume(job_description)
    
    # Save the resume
    print(f"Saving tailored resume to {output_path}...")
    tailor.tailor_resume(job_text=job_description, output_path=output_path)
    
    # Verify the resume meets all requirements
    print("Verifying resume meets all requirements...")
    results = verify_resume(tailored_resume, analysis)
    
    # Add warnings and substitutions to results
    results["warnings"] = tailor.warnings
    results["substitutions"] = tailor.substitutions
    
    # Print results
    print("\nTest Results:")
    print(f"Success: {results['success']}")
    print(f"Pages: {results['pages']}")
    print(f"Has all sections: {results['has_all_sections']}")
    print(f"Contains all required keywords: {results['contains_all_keywords']}")
    
    if results["missing_keywords"]:
        print("\nMissing Keywords:")
        for keyword in results["missing_keywords"]:
            print(f"- {keyword}")
    
    if results["substitutions"]:
        print("\nTechnology Substitutions:")
        for original, substitute in results["substitutions"].items():
            print(f"- {original} -> {substitute}")
    
    if results["warnings"]:
        print("\nWarnings:")
        for warning in results["warnings"]:
            print(f"- {warning}")
    
    return results


def verify_resume(resume_text: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify that a resume meets all requirements.
    
    Args:
        resume_text: The tailored resume text
        analysis: Job analysis dictionary
        
    Returns:
        A dictionary containing verification results
    """
    results = {
        "success": True,
        "pages": estimate_pages(resume_text),
        "has_all_sections": True,
        "contains_all_keywords": True,
        "missing_keywords": [],
    }
    
    # Check if the resume has all required sections
    required_sections = ["PROFESSIONAL SUMMARY", "SKILLS", "WORK EXPERIENCE", "EDUCATION"]
    for section in required_sections:
        if f"# {section}" not in resume_text:
            results["has_all_sections"] = False
            results["success"] = False
            break
    
    # Check if the resume is within the page limit
    if results["pages"] > 2:
        results["success"] = False
    
    # Check if the resume contains all required keywords
    important_keywords = analysis.get("keywords", []) + analysis.get("tech_stack", [])
    for keyword in important_keywords:
        if keyword.lower() not in resume_text.lower():
            results["contains_all_keywords"] = False
            results["missing_keywords"].append(keyword)
    
    if results["missing_keywords"]:
        results["success"] = False
    
    return results


if __name__ == "__main__":
    # Get command line arguments
    job_file = sys.argv[1] if len(sys.argv) > 1 else "example_job.txt"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "test_resume.docx"
    cv_dir = sys.argv[3] if len(sys.argv) > 3 else "cv"
    llm_provider = sys.argv[4] if len(sys.argv) > 4 else "openai"
    
    # Run the test
    results = test_resume_tailoring(job_file, output_path, cv_dir, llm_provider)
    
    # Save results to a file
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTest results saved to test_results.json")
    
    # Exit with appropriate status code
    sys.exit(0 if results["success"] else 1)
