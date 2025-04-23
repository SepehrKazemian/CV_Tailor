import logging
import logging
import json
from typing import List, Optional
from pydantic import ValidationError # Import for validation error handling

from resume_tailor.utils.llm_provider import get_llm
from resume_tailor.utils.llm_utils import run_llm_chain # Use the available function
# Import utility to extract JSON block, in case LLM wraps output
from resume_tailor.tailors.tailor_utils import extract_json_block
from resume_tailor.tailors.skill_graph.schema import ExtractedSkills

logger = logging.getLogger(__name__)

# Define the prompt for skill extraction
SKILL_EXTRACTION_PROMPT = """
Extract all specific technical skills, tools, libraries, frameworks, platforms, methodologies, and concepts from the following job description.
Focus on concrete nouns or noun phrases representing capabilities. Avoid generic terms like "communication" or "teamwork" unless explicitly listed as a required technical skill (e.g., "Agile Methodologies").

Return ONLY a JSON object containing a single key "skills" whose value is a list of the extracted skill strings.
Example: {{"skills": ["Python", "TensorFlow", "AWS S3", "Docker"]}}

Job Description:
{job_description}
"""

class SkillExtractor:
    """Extracts L1 skills from job description text using an LLM."""

    def __init__(self, llm_provider: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initializes the extractor.

        Args:
            llm_provider: LLM provider name (uses config if None).
            api_key: API key (uses credentials/env var if None).
        """
        # Note: We might want a shared LLM instance across modules later
        self.model_name, self.llm, _ = get_llm(llm_provider, api_key, return_raw=False)
        logger.info(f"Initialized SkillExtractor with model: {self.model_name}")

    def extract_skills(self, job_description: str) -> List[str]:
        """
        Extracts L1 skills from the provided job description text.

        Args:
            job_description: The job description text.

        Returns:
            A list of extracted skill strings, or an empty list on failure.
        """
        if not job_description:
            logger.warning("Job description is empty. Cannot extract skills.")
            return []

        logger.debug("Attempting to extract skills from job description...")
        try:
            # Use run_llm_chain to get the raw string output
            result_str = run_llm_chain(
                llm=self.llm,
                template_str=SKILL_EXTRACTION_PROMPT,
                input_vars=["job_description"],
                inputs={"job_description": job_description},
                fail_softly=True # Returns error string on failure
            )

            if result_str.startswith("[LLM FAILED]"):
                logger.error(f"Skill extraction LLM call failed: {result_str}")
                return []

            # Extract JSON block from the potentially messy LLM output
            json_str = extract_json_block(result_str)
            if not json_str:
                logger.error(f"Could not extract JSON block from LLM output. Raw: {result_str}")
                return []

            # Parse the JSON string
            parsed_data = json.loads(json_str)

            # --- Robust Parsing Logic ---
            extracted_skills_list = []
            if isinstance(parsed_data, dict):
                # Handle case where skills are directly under the 'skills' key
                if "skills" in parsed_data and isinstance(parsed_data["skills"], list):
                    extracted_skills_list.extend(item for item in parsed_data["skills"] if isinstance(item, str))
                # Handle case where skills are grouped under various keys (like the error case)
                else:
                    logger.warning(f"LLM returned structured JSON instead of simple skills list. Extracting from all list values.")
                    for key, value in parsed_data.items():
                        if isinstance(value, list):
                            # Assume list items are skills
                            extracted_skills_list.extend(item for item in value if isinstance(item, str))
            elif isinstance(parsed_data, list): # Handle case where LLM returns just a list
                 logger.warning(f"LLM returned a direct list instead of JSON object.")
                 extracted_skills_list.extend(item for item in parsed_data if isinstance(item, str))
            else:
                 logger.error(f"Unexpected format received from LLM after JSON parsing: {type(parsed_data)}")
                 return []

            # Validate that we got *some* skills
            if extracted_skills_list:
                 # Normalize skills: lowercase, strip whitespace, remove duplicates
                normalized_skills = sorted(list(set(skill.strip().lower() for skill in extracted_skills_list if skill.strip())))
                logger.info(f"Successfully extracted and normalized {len(normalized_skills)} unique skills.")
                logger.debug(f"Extracted skills: {normalized_skills}")
                return normalized_skills

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM output: {e}. JSON string: '{json_str}'")
            return []
        except ValidationError as e:
            logger.error(f"Failed to validate extracted skills against schema: {e}. Parsed data: {parsed_data}")
            return []
        except Exception as e:
            logger.error(f"An error occurred during skill extraction: {e}", exc_info=True)
            return []

# --- Test Execution ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Sample Job Description (replace with a real one for better testing)
    sample_jd = """
    We are seeking a Senior Machine Learning Engineer to join our dynamic team.
    Responsibilities include designing, developing, and deploying machine learning models using Python and TensorFlow.
    Experience with AWS services (S3, SageMaker) and Docker is required.
    Familiarity with NLP techniques and libraries like spaCy or NLTK is a plus.
    Must have strong experience with SQL databases and data warehousing concepts.
    Knowledge of CI/CD pipelines using Jenkins or GitLab CI is essential.
    Experience with Langchain and vector databases like Pinecone is highly desirable.
    """

    logger.info("--- Running SkillExtractor Test ---")
    try:
        # Initialize extractor (will use config/credentials/env for LLM/key)
        extractor = SkillExtractor()

        extracted_skills = extractor.extract_skills(sample_jd)

        if extracted_skills:
            print("\nExtracted L1 Skills:")
            for skill in extracted_skills:
                print(f"- {skill}")
        else:
            print("\nSkill extraction failed or no skills found. Check logs.")

        logger.info("--- SkillExtractor Test Finished ---")

    except Exception as e:
        logger.error(f"An unexpected error occurred during the test: {e}", exc_info=True)
