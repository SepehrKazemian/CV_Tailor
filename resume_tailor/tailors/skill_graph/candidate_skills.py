# resume_tailor/tailors/skill_graph/candidate_skills.py
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

# Assuming SkillPreprocessor is in the same directory or adjust import
from resume_tailor.tailors.skill_graph.skill_preprocessing import SkillPreprocessor

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

    def _calculate_file_hash(self, filepath: Path) -> str:
        """Calculates the SHA256 hash of a file's content."""
        hasher = hashlib.sha256()
        try:
            with open(filepath, 'rb') as file:
                while chunk := file.read(4096):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except FileNotFoundError:
            logger.warning(f"File not found for hashing: {filepath}")
            return ""
        except Exception as e:
            logger.error(f"Error calculating hash for {filepath}: {e}", exc_info=True)
            return ""

    def _check_file_changed(self) -> bool:
        """Checks if the file content has changed since the last stored hash."""
        if not self.skills_filepath.exists():
            logger.warning(f"Skills file to check does not exist: {self.skills_filepath}")
            # If file doesn't exist but hash does, consider it changed (to clear hash)
            return self.hash_filepath.exists()

        current_hash = self._calculate_file_hash(self.skills_filepath)
        if not current_hash: return True # Treat hash error as change

        stored_hash = ""
        if self.hash_filepath.exists():
            try:
                with open(self.hash_filepath, 'r') as f:
                    stored_hash = f.read().strip()
            except Exception as e:
                logger.error(f"Error reading hash file {self.hash_filepath}: {e}", exc_info=True)
                return True # Treat read error as change

        if current_hash != stored_hash:
            logger.info(f"Change detected in {self.skills_filepath} (hash mismatch).")
            try:
                self.hash_filepath.parent.mkdir(parents=True, exist_ok=True)
                with open(self.hash_filepath, 'w') as f:
                    f.write(current_hash)
                logger.info(f"Updated hash stored in {self.hash_filepath}")
            except Exception as e:
                 logger.error(f"Error writing hash file {self.hash_filepath}: {e}", exc_info=True)
            return True
        else:
            logger.debug(f"No changes detected in {self.skills_filepath} based on hash.")
            return False

    def _parse_skills_file(self) -> List[str]:
        """Parses skills from the file, handling potential formatting."""
        skills = []
        if not self.skills_filepath.exists():
            logger.error(f"Skills file not found: {self.skills_filepath}")
            return []
        try:
            with open(self.skills_filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    if line.startswith(('•', '*', '-', '+')):
                        line = line[1:].strip()
                    if ':' in line:
                        parts = line.split(':', 1)
                        # header = parts[0].strip() # Ignore header for now
                        skill_part = parts[1].strip()
                        skills.extend([s.strip() for s in skill_part.split(',') if s.strip()])
                    else:
                         skills.extend([s.strip() for s in line.split(',') if s.strip()])
            unique_skills = sorted(list(set(s for s in skills if s))) # Ensure unique and non-empty
            logger.info(f"Parsed {len(unique_skills)} unique raw skills from {self.skills_filepath}.")
            return unique_skills
        except Exception as e:
            logger.error(f"Error parsing skills file {self.skills_filepath}: {e}", exc_info=True)
            return []

    def process_skills(self, force_process: bool = False) -> None:
        """
        Checks for file changes, parses, standardizes, and stores candidate skills.

        Args:
            force_process: If True, process the file even if the hash hasn't changed.
        """
        file_changed = self._check_file_changed()

        if force_process or file_changed or not self._processed:
            logger.info(f"Processing candidate skills file: {self.skills_filepath} (Force: {force_process}, Changed: {file_changed}, FirstRun: {not self._processed})")
            self.raw_skills.clear()
            self.standardized_skills.clear()
            self.standardization_map.clear() # Clear the map on re-process

            parsed_skills = self._parse_skills_file()
            if parsed_skills:
                self.raw_skills.update(parsed_skills)
                # Run standardization
                std_skills_list, current_map = self.preprocessor.run(parsed_skills)
                self.standardized_skills.update(std_skills_list)
                self.standardization_map = current_map # Store the latest map
                logger.info(f"Stored {len(self.standardized_skills)} standardized candidate skills and {len(self.standardization_map)} mappings.")
            else:
                logger.warning(f"Skill file {self.skills_filepath} was empty or could not be parsed. No candidate skills loaded.")
            self._processed = True # Mark as processed
        else:
             logger.info(f"Skipping processing of {self.skills_filepath} as it hasn't changed and has already been processed.")

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
        if not self._processed or force_process:
            self.process_skills(force_process=force_process)
        return self.raw_skills, self.standardized_skills, self.standardization_map
