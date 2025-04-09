import logging
import os
from typing import Optional

# Moved imports to the top
from resume_tailor.analyzers.job_analyzer import JobAnalyzer
from resume_tailor.tailors.professional_summary_tailor import ProfessionalSummaryTailor
from resume_tailor.utils.file_utils import read_job_description, read_resume_components
from resume_tailor.utils.llm_provider import get_llm  # Needed to check provider validity

# Configure root logger to see logs from modules
# Basic config is fine here, professional_summary_tailor configures its own file handler
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Configuration ---
LLM_PROVIDER = "claude"  # Or load from config.py / env var
API_KEY: Optional[str] = None  # Set if not using env var or config.py
CV_DIR = "cv"
JOB_DESCRIPTION_FILE = "example_job.txt"
# --- End Configuration ---


def main():
    """Runs the professional summary refinement test."""
    logger = logging.getLogger(__name__)  # Define logger within main scope
    logger.info("--- Starting Professional Summary Refinement Test ---")

    # Validate LLM Provider
    try:
        get_llm(LLM_PROVIDER, API_KEY)
        logger.info(f"Using LLM Provider: {LLM_PROVIDER}")
    except ValueError as e:
        logger.error(f"Invalid LLM provider specified: {e}")
        return
    except Exception as e:
        # Catch potential API key errors early if get_llm tries to instantiate
        logger.error(f"Error initializing LLM provider (check API key?): {e}")
        return

    # 1. Read Inputs
    logger.info(f"Reading job description from: {JOB_DESCRIPTION_FILE}")
    job_description = read_job_description(JOB_DESCRIPTION_FILE)
    if not job_description:
        logger.error("Failed to read job description.")
        return

    logger.info(f"Reading original resume components from: {CV_DIR}")
    resume_components = read_resume_components(CV_DIR)
    original_summary = resume_components.get("professional_summary")
    if not original_summary:
        logger.error("Failed to read original professional summary from cv/ps.txt.")
        return
    logger.info("Original Professional Summary:\n" + original_summary)

    # 2. Analyze Job Description
    logger.info("Initializing Job Analyzer...")
    try:
        job_analyzer = JobAnalyzer(llm_provider=LLM_PROVIDER, api_key=API_KEY, cv_dir=CV_DIR)
        logger.info("Running job analysis...")
        analysis = job_analyzer.analyze(job_description)
        if not analysis or not isinstance(analysis, dict):
            logger.error("Job analysis returned invalid data.")
            return
        logger.info("Job analysis complete.")
        # logger.debug(f"Job Analysis Result: {analysis}")  # Optional: log full analysis
    except Exception as e:
        logger.error(f"Error during job analysis: {e}", exc_info=True)
        return

    # 3. Refine Professional Summary
    logger.info("Initializing Professional Summary Tailor...")
    try:
        summary_tailor = ProfessionalSummaryTailor(llm_provider=LLM_PROVIDER, api_key=API_KEY)
        # Access log file path from the instance if needed
        log_file_path = getattr(summary_tailor, 'log_file', 'professional_summary_refinement.log')
        logger.info("Running professional summary refinement loop...")
        final_summary = summary_tailor.refine_summary(
            analysis, original_summary, job_description
        )
        logger.info("--- Professional Summary Refinement Complete ---")
        print("\n" + "="*30)
        print(" Final Refined Professional Summary:")
        print("="*30)
        print(final_summary)
        print("="*30)
        print(f"\nDetailed logs available in: {log_file_path}")

    except Exception as e:
        logger.error(f"Error during summary refinement: {e}", exc_info=True)


if __name__ == "__main__":
    # Define logger for the __main__ block
    logger = logging.getLogger(__name__)

    # Ensure API key is set if needed (Example: using config.py)
    try:
        import config
        if LLM_PROVIDER == "claude" or LLM_PROVIDER == "anthropic":
            if 'ANTHROPIC_API_KEY' not in os.environ and hasattr(config, 'key'):
                os.environ['ANTHROPIC_API_KEY'] = config.key
        # Add similar checks for other providers if necessary
    except ImportError:
        logger.warning("config.py not found. Relying on environment variables for API keys.")
    except AttributeError:
        logger.warning("API key not found in config.py. Relying on environment variables.")

    main()
