#!/usr/bin/env python3
"""
Resume Tailoring Feedback Loop

This script implements a feedback loop for testing the resume tailoring pipeline.
It tailors resumes, evaluates them using an LLM-as-judge, and provides feedback for improvement.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path

from resume_tailor.tailors.resume_tailor import ResumeTailor
from resume_tailor.evaluators.ats_evaluator import ATSEvaluator
from resume_tailor.utils.file_utils import read_resume_components, read_job_description


def run_feedback_loop(job_description: str, 
                     resume_components: Dict[str, str],
                     cv_dir: str = "cv",
                     output_dir: str = "output",
                     iterations: int = 3,
                     llm_provider: Optional[str] = None,
                     api_key: Optional[str] = None,
                     format: str = "docx") -> None:
    """
    Run a feedback loop for testing the resume tailoring pipeline.
    
    Args:
        job_description: The job description text
        resume_components: The original resume components
        cv_dir: Directory containing resume component files
        output_dir: Directory to save the results
        iterations: Number of iterations to run
        llm_provider: LLM provider to use
        api_key: API key for the LLM provider
        format: Output format ('docx', 'txt', or 'md')
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the resume tailor
    tailor = ResumeTailor(cv_dir=cv_dir, llm_provider=llm_provider, api_key=api_key)
    # Override the resume components with the ones we read
    tailor.resume_components = resume_components
    
    # Initialize the ATS evaluator
    evaluator = ATSEvaluator(llm_provider=llm_provider, api_key=api_key)
    
    current_resume = None
    
    for i in range(iterations):
        print(f"\n=== Iteration {i+1}/{iterations} ===\n")
        
        # Save the job description
        job_file = os.path.join(output_dir, f"job_description.txt")
        with open(job_file, "w") as f:
            f.write(job_description)
        print(f"Job description saved to {job_file}")
        
        # Generate a tailored resume
        print("Generating tailored resume...")
        if i == 0:
            # First iteration: generate from scratch
            tailored_resume = tailor.generate_tailored_resume(job_description)
        else:
            # Subsequent iterations: use feedback to improve
            # We'll use the ATS evaluator's feedback to guide the improvement
            prompt_template = f"""
            You are a skilled resume writer. You need to improve this resume based on feedback.
            
            Original Resume:
            {current_resume}
            
            Feedback:
            {feedback}
            
            Improve the resume while maintaining the following structure:
            1. Professional Summary
            2. Skills
            3. Work Experience
            4. Education
            
            Make sure each section is clearly labeled and formatted properly.
            """
            
            # Use the LLM to improve the resume
            from langchain.chains import LLMChain
            from langchain.prompts import PromptTemplate
            from resume_tailor.utils.llm_provider import get_llm
            
            llm = get_llm(llm_provider, api_key)
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=[]
            )
            
            chain = LLMChain(llm=llm, prompt=prompt)
            tailored_resume = chain.run()
        
        current_resume = tailored_resume
        
        # Save the tailored resume
        resume_file = os.path.join(output_dir, f"tailored_resume_{i+1}.md")
        with open(resume_file, "w") as f:
            f.write(tailored_resume)
        print(f"Tailored resume saved to {resume_file}")
        
        # Save in the requested format
        if format != "md":
            format_file = os.path.join(output_dir, f"tailored_resume_{i+1}.{format}")
            
            # Save the resume in the requested format
            if format == "docx":
                from resume_tailor.output.docx_generator import generate_docx
                generate_docx(tailored_resume, format_file)
            else:
                # Save as plain text
                with open(format_file, 'w') as file:
                    file.write(tailored_resume)
            
            print(f"Tailored resume also saved as {format_file}")
        
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
        
        # Check if the resume has the correct outline
        has_correct_outline = all(
            section in tailored_resume.lower() 
            for section in ["professional summary", "skills", "experience", "education"]
        )
        
        if has_correct_outline and evaluation.ats_score >= 0.8:
            print(f"\nSuccess! Resume has the correct outline and a good ATS score ({evaluation.ats_score:.2f}).")
            break
        
        if i == iterations - 1:
            print("\nReached maximum iterations. Final resume may still need improvements.")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Run a feedback loop for testing the resume tailoring pipeline.')
    
    # Input options
    parser.add_argument('--job-file', type=str, default='example_job.txt', help='Path to a job description file')
    parser.add_argument('--cv-dir', type=str, default='cv', help='Directory containing resume components')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save the results')
    parser.add_argument('--format', type=str, choices=['docx', 'txt', 'md'], default='docx',
                        help='Output format (default: docx)')
    
    # LLM options
    parser.add_argument('--llm', type=str, choices=['openai', 'google', 'anthropic', 'claude'], 
                        default=None, help='LLM provider to use (default: use config.py)')
    parser.add_argument('--api-key', type=str, help='API key for the LLM provider')
    
    # Feedback loop options
    parser.add_argument('--iterations', type=int, default=3, help='Number of iterations to run')
    
    args = parser.parse_args()
    
    # Import config.py dynamically
    import importlib.util
    
    # Try to import config.py
    try:
        config_path = Path(__file__).resolve().parent / "config.py"
        spec = importlib.util.spec_from_file_location("config", config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
    except Exception:
        # If config.py can't be imported, use default values
        config = type('', (), {})()
        config.LLM = "anthropic"
        config.version = None
        config.key = None
    
    # If LLM is not specified, use the one from config.py
    if args.llm is None:
        args.llm = getattr(config, 'LLM', 'anthropic').lower()
    
    # Set API key if provided
    if args.api_key:
        if args.llm == 'openai':
            os.environ['OPENAI_API_KEY'] = args.api_key
        elif args.llm == 'google':
            os.environ['GOOGLE_API_KEY'] = args.api_key
        elif args.llm == 'anthropic' or args.llm == 'claude':
            os.environ['ANTHROPIC_API_KEY'] = args.api_key
    elif args.llm == 'anthropic' or args.llm == 'claude':
        # Use the API key from config.py
        api_key = getattr(config, 'key', None)
        if api_key:
            os.environ['ANTHROPIC_API_KEY'] = api_key
    
    # Read the resume components
    resume_components = read_resume_components(args.cv_dir)
    
    # Read the job description
    try:
        job_description = read_job_description(args.job_file)
    except FileNotFoundError:
        print(f"Error: Job description file {args.job_file} not found.")
        return 1
    
    # Run the feedback loop
    run_feedback_loop(
        job_description=job_description,
        resume_components=resume_components,
        cv_dir=args.cv_dir,
        output_dir=args.output_dir,
        iterations=args.iterations,
        llm_provider=args.llm,
        api_key=args.api_key,
        format=args.format
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
