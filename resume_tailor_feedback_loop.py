#!/usr/bin/env python3
"""
Resume Tailoring Feedback Loop

This script implements a feedback loop for testing the resume tailoring pipeline.
It generates random job descriptions, tailors resumes, evaluates them using an LLM-as-judge,
and provides feedback for improvement.
"""

import os
import sys
import json
import argparse
import random
from typing import Dict, List, Any, Optional, Set
from pathlib import Path

from resume_tailor.tailors.resume_tailor import ResumeTailor
from resume_tailor.evaluators.ats_evaluator import ATSEvaluator, ResumeEvaluation
from resume_tailor.evaluators.job_generator import JobDescriptionGenerator
from resume_tailor.utils.file_utils import read_resume_components, read_job_description


def run_feedback_loop(job_description: str, 
                     resume_components: Dict[str, str],
                     output_dir: str = "feedback_results",
                     iterations: int = 1,
                     llm_provider: Optional[str] = None,
                     api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Run a feedback loop for testing the resume tailoring pipeline.
    
    Args:
        job_description: The job description text
        resume_components: The original resume components
        output_dir: Directory to save the results
        iterations: Number of iterations to run
        llm_provider: LLM provider to use
        api_key: API key for the LLM provider
        
    Returns:
        List of results for each iteration
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the resume tailor
    tailor = ResumeTailor(cv_dir=None, llm_provider=llm_provider, api_key=api_key)
    tailor.resume_components = resume_components
    
    # Initialize the ATS evaluator
    evaluator = ATSEvaluator(llm_provider=llm_provider, api_key=api_key)
    
    results = []
    
    for i in range(iterations):
        print(f"\n=== Iteration {i+1}/{iterations} ===\n")
        
        # Save the job description
        job_file = os.path.join(output_dir, f"job_description_{i+1}.txt")
        with open(job_file, "w") as f:
            f.write(job_description)
        print(f"Job description saved to {job_file}")
        
        # Generate a tailored resume
        print("Generating tailored resume...")
        tailored_resume = tailor.generate_tailored_resume(job_description)
        
        # Save the tailored resume
        resume_file = os.path.join(output_dir, f"tailored_resume_{i+1}.md")
        with open(resume_file, "w") as f:
            f.write(tailored_resume)
        print(f"Tailored resume saved to {resume_file}")
        
        # Evaluate the resume
        print("Evaluating resume...")
        evaluation = evaluator.evaluate_resume(
            job_description=job_description,
            resume_text=tailored_resume,
            original_resume_components=resume_components
        )
        
        # Save the evaluation
        evaluation_file = os.path.join(output_dir, f"evaluation_{i+1}.json")
        evaluation.save_to_file(evaluation_file)
        print(f"Evaluation saved to {evaluation_file}")
        
        # Print the evaluation summary
        print("\nEvaluation Summary:")
        evaluation.print_summary()
        
        # Get improvement feedback
        print("\nGetting improvement feedback...")
        feedback = evaluator.get_improvement_feedback(
            evaluation=evaluation,
            job_description=job_description,
            resume_text=tailored_resume
        )
        
        # Save the feedback
        feedback_file = os.path.join(output_dir, f"feedback_{i+1}.txt")
        with open(feedback_file, "w") as f:
            f.write(feedback)
        print(f"Improvement feedback saved to {feedback_file}")
        
        # Print the feedback
        print("\nImprovement Feedback:")
        print(feedback)
        
        # Store the results
        results.append({
            "iteration": i+1,
            "job_description_file": job_file,
            "tailored_resume_file": resume_file,
            "evaluation_file": evaluation_file,
            "feedback_file": feedback_file,
            "ats_score": evaluation.ats_score,
            "missing_skills": evaluation.missing_skills,
            "substitutions": evaluation.substitutions
        })
        
        # If there are more iterations, generate a new job description
        if i < iterations - 1 and iterations > 1:
            print("\nGenerating new job description for next iteration...")
            job_generator = JobDescriptionGenerator(llm_provider=llm_provider, api_key=api_key)
            resume_skills = job_generator.extract_skills_from_resume(resume_components)
            job_description = job_generator.generate_job_description_with_resume_skills(resume_skills)
    
    # Save the overall results
    results_file = os.path.join(output_dir, "feedback_loop_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nOverall results saved to {results_file}")
    
    return results


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Run a feedback loop for testing the resume tailoring pipeline.')
    
    # Input options
    parser.add_argument('--job-file', type=str, help='Path to a job description file')
    parser.add_argument('--job-text', type=str, help='Job description text')
    parser.add_argument('--cv-dir', type=str, default='cv', help='Directory containing resume components')
    parser.add_argument('--random-job', action='store_true', help='Generate a random job description')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='feedback_results', help='Directory to save the results')
    
    # LLM options
    parser.add_argument('--llm', type=str, choices=['openai', 'google', 'anthropic', 'claude'], 
                        help='LLM provider to use (default: use config.py)')
    parser.add_argument('--api-key', type=str, help='API key for the LLM provider')
    
    # Feedback loop options
    parser.add_argument('--iterations', type=int, default=1, help='Number of iterations to run')
    
    args = parser.parse_args()
    
    # Read the resume components
    resume_components = read_resume_components(args.cv_dir)
    
    # Get the job description
    job_description = None
    
    if args.random_job:
        # Generate a random job description
        print("Generating random job description...")
        job_generator = JobDescriptionGenerator(llm_provider=args.llm, api_key=args.api_key)
        resume_skills = job_generator.extract_skills_from_resume(resume_components)
        job_description = job_generator.generate_job_description_with_resume_skills(resume_skills)
    elif args.job_file:
        # Read the job description from a file
        try:
            job_description = read_job_description(args.job_file)
        except FileNotFoundError:
            print(f"Error: Job description file {args.job_file} not found.")
            return 1
    elif args.job_text:
        # Use the provided job description text
        job_description = args.job_text
    else:
        # Use the example job description
        example_job_file = "example_job.txt"
        if os.path.exists(example_job_file):
            job_description = read_job_description(example_job_file)
            print(f"Using example job description from {example_job_file}")
        else:
            print("Error: No job description provided and example_job.txt not found.")
            print("Please provide a job description using --job-file, --job-text, or --random-job.")
            return 1
    
    # Run the feedback loop
    run_feedback_loop(
        job_description=job_description,
        resume_components=resume_components,
        output_dir=args.output_dir,
        iterations=args.iterations,
        llm_provider=args.llm,
        api_key=args.api_key
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
