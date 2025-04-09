import os
import sys
import argparse
import tempfile
import logging
from typing import Optional

from resume_tailor.tests.test_pipeline import test_resume_tailoring
from resume_tailor.tailors.resume_tailor import ResumeTailor
from resume_tailor.utils.llm_provider import get_provider_env_var

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Tailor your resume based on a job description.')

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--job-file', type=str, help='Path to a file containing the job description')
    input_group.add_argument('--job-text', type=str, help='Job description text')

    parser.add_argument('--output', type=str, default='tailored_resume.docx',
                        help='Output file path (default: tailored_resume.docx)')
    parser.add_argument('--format', type=str, choices=['docx', 'txt', 'md'], default='docx',
                        help='Output format (default: docx)')
    parser.add_argument('--llm', type=str, choices=['openai', 'google', 'anthropic'], default='openai',
                        help='LLM provider to use (default: openai)')
    parser.add_argument('--api-key', type=str, help='API key for the selected LLM provider')
    parser.add_argument('--cv-dir', type=str, default='cv',
                        help='Directory containing resume component files (default: cv)')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode to verify the resume meets all requirements')

    return parser.parse_args()

def set_api_key(llm_provider: str, api_key: Optional[str]) -> bool:
    """
    Set the API key for the selected LLM provider.

    Args:
        llm_provider (str): LLM provider name.
        api_key (Optional[str]): API key string.

    Returns:
        bool: True if API key is set, False otherwise.
    """
    api_key_env_var = get_provider_env_var(llm_provider)
    if api_key:
        os.environ[api_key_env_var] = api_key
        logger.info(f"API key set for {llm_provider}")
    return api_key_env_var in os.environ

def ensure_output_extension(output_path: str, fmt: str) -> str:
    """
    Ensure the output file path ends with the correct extension.

    Args:
        output_path (str): The user-provided output file path.
        fmt (str): The desired file format (e.g., 'docx', 'txt', 'md').

    Returns:
        str: Updated output path with the correct extension.
    """
    if not output_path.endswith(f'.{fmt}'):
        output_path = f"{output_path.rsplit('.', 1)[0] if '.' in output_path else output_path}.{fmt}"
        logger.info(f"Updated output path to: {output_path}")
    return output_path

def run_test_mode(args, output_path: str) -> int:
    """
    Run the resume tailoring in test mode.

    Args:
        args (argparse.Namespace): Parsed arguments.
        output_path (str): Final output file path.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    logger.info("Running in test mode...")
    job_file = args.job_file
    if args.job_text:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp:
            temp.write(args.job_text)
            job_file = temp.name
            logger.info(f"Temporary job file created at: {job_file}")

    try:
        results = test_resume_tailoring(job_file=job_file, output_path=output_path,
                                        cv_dir=args.cv_dir, llm_provider=args.llm)
        return 0 if results["success"] else 1
    finally:
        if args.job_text:
            os.unlink(job_file)
            logger.info(f"Temporary job file deleted: {job_file}")

def run_normal_mode(args, output_path: str) -> int:
    """
    Run the resume tailoring in normal mode.

    Args:
        args (argparse.Namespace): Parsed arguments.
        output_path (str): Final output file path.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    tailor = ResumeTailor(cv_dir=args.cv_dir, llm_provider=args.llm)

    logger.info(f"Analyzing job description using {args.llm}...")
    tailor.tailor_resume(
        job_file=args.job_file,
        job_text=args.job_text,
        output_path=output_path,
        format=args.format
    )

    logger.info(f"Tailored resume saved to: {output_path}")

    if tailor.warnings:
        logger.warning("Warnings detected:")
        for warning in tailor.warnings:
            logger.warning(f"- {warning}")

    if tailor.substitutions:
        logger.info("Technology Substitutions:")
        for original, substitute in tailor.substitutions.items():
            logger.info(f"- {original} -> {substitute}")

    return 0

def main():
    """
    Main function to run the CLI.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    args = parse_arguments()

    if not set_api_key(args.llm, args.api_key):
        logger.error(f"{args.llm.capitalize()} API key not found. Please set it using --api-key or the "
                     f"{get_provider_env_var(args.llm)} environment variable.")
        return 1

    output_path = ensure_output_extension(args.output, args.format)

    try:
        if args.test:
            return run_test_mode(args, output_path)
        else:
            return run_normal_mode(args, output_path)
    except Exception as e:
        logger.exception(f"Error tailoring resume: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
