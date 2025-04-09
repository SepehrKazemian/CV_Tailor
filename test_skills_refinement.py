import logging
import os
from typing import Optional

# Configure root logger to see logs from modules
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import the new SkillsRefiner
from resume_tailor.tailors.skills_tailor import SkillsRefiner
from resume_tailor.utils.file_utils import read_job_description, read_resume_components
from resume_tailor.utils.llm_provider import get_llm # Needed to check provider validity

# --- Configuration ---
LLM_PROVIDER = "claude" # Or load from config.py / env var
API_KEY: Optional[str] = None # Set if not using env var or config.py
CV_DIR = "cv"
JOB_DESCRIPTION_FILE = "example_job.txt"
# --- End Configuration ---

def main():
    """Runs the skills section refinement test."""
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Skills Section Refinement Test ---")

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

    logger.info(f"Reading original resume components from: {CV_DIR}")
    resume_components = read_resume_components(CV_DIR)
    original_skills = resume_components.get("skills")
    if not original_skills:
        logger.error("Failed to read original skills from cv/skills.txt.")
        return
    logger.info("Original Skills Text:\n" + original_skills)

    # 2. Refine Skills Section
    logger.info("Initializing Skills Refiner...")
    try:
        skills_refiner = SkillsRefiner(llm_provider=LLM_PROVIDER, api_key=API_KEY)
        # Access log file path from the instance if needed
        log_file_path = getattr(skills_refiner, 'log_file', 'skills_refinement.log')
        logger.info("Running skills refinement loop...")
        final_skills_section = skills_refiner.refine_skills(
            job_description, original_skills
        )
        logger.info("--- Skills Section Refinement Complete ---")
        print("\n" + "="*30)
        print(" Final Refined Skills Section:")
        print("="*30)
        print(final_skills_section)
        print("="*30)
        print(f"\nDetailed logs available in: {log_file_path}")

    except Exception as e:
        logger.error(f"Error during skills refinement: {e}", exc_info=True)

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
