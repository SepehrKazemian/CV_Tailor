import logging
import os
from typing import Optional

# Configure root logger to see logs from modules
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import the new ExperienceRefiner and necessary utilities
from resume_tailor.tailors.experience_tailor import ExperienceRefiner
from resume_tailor.analyzers.job_analyzer import JobAnalyzer # Needed for analysis input
from resume_tailor.utils.file_utils import read_job_description, read_experience_yaml
from resume_tailor.utils.llm_provider import get_llm # Needed to check provider validity

# --- Configuration ---
LLM_PROVIDER = "claude" # Or load from config.py / env var
API_KEY: Optional[str] = None # Set if not using env var or config.py
CV_DIR = "cv"
EXPERIENCE_FILE = os.path.join(CV_DIR, "experience.yml")
JOB_DESCRIPTION_FILE = "example_job.txt"
# --- End Configuration ---

def main():
    """Runs the experience section refinement test."""
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Experience Section Refinement Test ---")

    # Validate LLM Provider
    try:
        get_llm(LLM_PROVIDER, API_KEY)
        logger.info(f"Using LLM Provider: {LLM_PROVIDER}")
    except ValueError as e:
        logger.error(f"Invalid LLM provider specified: {e}")
        return
    except Exception as e:
         logger.error(f"Error initializing LLM provider (check API key?): {e}")
         return

    # 1. Read Inputs
    logger.info(f"Reading job description from: {JOB_DESCRIPTION_FILE}")
    job_description = read_job_description(JOB_DESCRIPTION_FILE)
    if not job_description:
        logger.error("Failed to read job description.")
        return

    logger.info(f"Reading original experience from: {EXPERIENCE_FILE}")
    original_experience_data = read_experience_yaml(EXPERIENCE_FILE)
    if not original_experience_data:
        logger.error("Failed to read or parse experience YAML.")
        return
    logger.info("Original Experience Data Loaded.")
    # logger.debug(f"Original Experience YAML data:\n{original_experience_data}")

    # 2. Analyze Job Description (needed for context in refinement)
    logger.info("Initializing Job Analyzer...")
    try:
        # Note: JobAnalyzer needs cv_dir to load its own keyword caches
        job_analyzer = JobAnalyzer(llm_provider=LLM_PROVIDER, api_key=API_KEY, cv_dir=CV_DIR)
        logger.info("Running job analysis...")
        analysis = job_analyzer.analyze(job_description)
        if not analysis or not isinstance(analysis, dict):
             logger.error("Job analysis returned invalid data.")
             return
        logger.info("Job analysis complete.")
        # logger.debug(f"Job Analysis Result: {analysis}")
    except Exception as e:
        logger.error(f"Error during job analysis: {e}", exc_info=True)
        return

    # 3. Refine Experience Section
    logger.info("Initializing Experience Refiner...")
    try:
        experience_refiner = ExperienceRefiner(llm_provider=LLM_PROVIDER, api_key=API_KEY)
        # Access log file path from the instance if needed
        log_file_path = getattr(experience_refiner, 'log_file', 'experience_refinement.log')
        logger.info("Running experience refinement loop...")
        final_experience_section = experience_refiner.refine_experience(
            original_experience_data, job_description, analysis
        )
        logger.info("--- Experience Section Refinement Complete ---")
        print("\n" + "="*30)
        print(" Final Refined Experience Section:")
        print("="*30)
        print(final_experience_section)
        print("="*30)
        print(f"\nDetailed logs available in: {log_file_path}")

    except Exception as e:
        logger.error(f"Error during experience refinement: {e}", exc_info=True)

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
