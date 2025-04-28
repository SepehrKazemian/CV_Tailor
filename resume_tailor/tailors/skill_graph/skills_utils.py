# resume_tailor/tailors/skills/skills_utils.py
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

def parse_skills_file(filepath: Path) -> List[str]:
    """
    Parses skills from a file, handling potential formatting like bullets and headers.

    Args:
        filepath: Path to the skills file.

    Returns:
        A list of unique, non-empty skill strings found in the file.
    """
    skills = []
    if not filepath.exists():
        logger.error(f"Skills file not found: {filepath}")
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                # Handle lines starting with bullets or similar markers
                if line.startswith(('•', '*', '-', '+')):
                    line = line[1:].strip()
                # Split potentially comma-separated skills within a line, considering headers
                if ':' in line:
                    # Assume format like "Header: Skill1, Skill2"
                    parts = line.split(':', 1)
                    # header = parts[0].strip() # Ignore header for now
                    skill_part = parts[1].strip()
                    skills.extend([s.strip() for s in skill_part.split(',') if s.strip()])
                else:
                    # Treat the whole line as one or more comma-separated skills
                     skills.extend([s.strip() for s in line.split(',') if s.strip()])
        # Ensure unique and non-empty skills
        unique_skills = sorted(list(set(s for s in skills if s)))
        logger.info(f"Parsed {len(unique_skills)} unique raw skills from {filepath}.")
        return unique_skills
    except Exception as e:
        logger.error(f"Error parsing skills file {filepath}: {e}", exc_info=True)
        return []
