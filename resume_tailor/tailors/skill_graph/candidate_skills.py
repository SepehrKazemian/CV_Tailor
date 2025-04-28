# resume_tailor/tailors/skill_graph/candidate_skills.py
import logging
# import hashlib # Removed
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

# Import the utility function for checking changes
from resume_tailor.utils.check_changes_hash import check_file_changed
# Import the new parsing utility function
from resume_tailor.tailors.skills.skills_utils import parse_skills_file
# Assuming SkillPreprocessor is in the same directory or adjust import
from resume_tailor.tailors.skill_graph.skill_preprocessing import SkillPreprocessor
from resume_tailor.tailors.skill_graph.skill_graph_llms import skill_domain_classifier


logger = logging.getLogger(__name__)

class CandidateSkillProcessor:
    """Handles reading, parsing, and standardizing skills from the candidate skills file."""

    def __init__(self, skills_filepath: str = "cv/skills.txt", llm_provider: str = 'anthropic'):
        """
        Initializes the processor.

        Args:
            skills_filepath: Path to the candidate skills file.
            llm_provider: LLM provider to use for standardization.
        """
        self.skills_filepath = Path(skills_filepath)
        self.hash_filepath = self.skills_filepath.parent / f".{self.skills_filepath.stem}_hash" # e.g., cv/.skills_hash
        self.preprocessor = SkillPreprocessor(llm_provider=llm_provider)
        self.raw_skills: Set[str] = set()
        self.standardized_skills: Set[str] = set()
        self.standardization_map: Dict[str, Optional[str]] = {}
        self._processed = False # Flag to track if processing has occurred

    # Removed _calculate_file_hash and _check_file_changed methods
    # Removed _parse_skills_file method

    def process_skills(self, force_process: bool = False) -> None:
        """
        Checks for file changes, parses, standardizes, and stores candidate skills.

        Args:
            force_process: If True, process the file even if the hash hasn't changed.
        """
        # Use the imported utility function here
        file_changed = check_file_changed(self.skills_filepath, self.hash_filepath)

        # Logic to decide whether to process based on change/force/processed flag
        if force_process or file_changed or not self._processed:
            logger.info(f"Processing candidate skills file: {self.skills_filepath} (Force: {force_process}, Changed: {file_changed}, FirstRun: {not self._processed})")
            self.raw_skills.clear()
            self.standardized_skills.clear()
            self.standardization_map.clear() # Clear the map on re-process

            # Use the imported parse_skills_file utility
            parsed_skills = parse_skills_file(self.skills_filepath)
            if parsed_skills:
                self.raw_skills.update(parsed_skills)
                # Run standardization
                std_skills_list, current_map = self.preprocessor.run(parsed_skills) # Assuming run returns list, map
                self.standardized_skills.update(std_skills_list)
                self.standardization_map = current_map # Store the latest map
                logger.info(f"Stored {len(self.standardized_skills)} standardized candidate skills and {len(self.standardization_map)} mappings.")
                self._processed = True # Mark as processed
            else:
                logger.warning(f"Skill file {self.skills_filepath} was empty or could not be parsed. No candidate skills loaded.")
                self._processed = True # Mark as processed even if empty to avoid reprocessing unless forced/changed
        else:
             logger.info(f"Skipping processing of {self.skills_filepath} as it hasn't changed and has already been processed.")

        # This method doesn't need to return these values directly anymore
        # return std_skills_list, current_map, soft_skills # Removed return

    def get_skills(self, force_process: bool = False) -> Tuple[Set[str], Set[str], Dict[str, Optional[str]]]:
        """
        Ensures skills are processed (if needed) and returns them.

        Args:
            force_process: If True, force reprocessing the file.

        Returns:
            Tuple containing:
            - Set of raw skills.
            - Set of standardized skills.
            - Dictionary mapping raw skills to standardized skills (or None).
        """
        # Ensure processing has happened if needed
        if not self._processed or force_process:
             # Call process_skills which now handles the check internally
             self.process_skills(force_process=force_process)

        return self.raw_skills, self.standardized_skills, self.standardization_map



