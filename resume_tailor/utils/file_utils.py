import os
import json
import logging
from typing import Dict, List, Optional, Set # Added Set
import os
import json
import logging
from typing import Dict, List, Optional, Set
import yaml
import re
import hashlib # For change detection
from pathlib import Path # For path manipulation

logger = logging.getLogger(__name__)

# --- YAML Reading ---
def read_experience_yaml(filepath: str) -> Optional[Dict]:
    """Reads and parses the experience YAML file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            experience_data = yaml.safe_load(file)
            if not isinstance(experience_data, dict):
                 logger.error(f"Experience YAML file did not parse into a dictionary: {filepath}")
                 return None
            return experience_data
    except FileNotFoundError:
        logger.error(f"Experience YAML file not found: {filepath}")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing experience YAML file {filepath}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error reading experience YAML file {filepath}: {e}")
        return None

# --- Component Reading ---
def read_resume_components(cv_dir: str) -> Dict[str, any]:
    """
    Read resume components from files in the specified directory.
    Reads .txt files for summary, skills, education.
    Reads experience.yml for experience.
    """
    components = {}
    txt_files = {
        "professional_summary": "ps.txt",
        "skills": "skills.txt",
        "education": "education.txt"
    }

    for key, filename in txt_files.items():
        filepath = os.path.join(cv_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                components[key] = file.read().strip()
        except FileNotFoundError:
            logger.warning(f"Component file not found: {filepath}. Setting empty string for '{key}'.")
            components[key] = ""
        except Exception as e:
            logger.error(f"Error reading component file {filepath}: {e}")
            components[key] = "" # Default to empty on error

    # Read experience from YAML
    experience_filepath = os.path.join(cv_dir, "experience.yml")
    experience_data = read_experience_yaml(experience_filepath)
    components["experience"] = experience_data if experience_data else {} # Store dict or empty dict

    return components

# --- Job Description Reading ---
def read_job_description(job_file: Optional[str] = None, job_text: Optional[str] = None) -> Optional[str]:
    """Reads job description from file or text."""
    if job_text:
        return job_text.strip()
    elif job_file:
        try:
            with open(job_file, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except FileNotFoundError:
            logger.error(f"Job description file not found: {job_file}")
            return None
        except Exception as e:
            logger.error(f"Error reading job description file {job_file}: {e}")
            return None
    else:
        logger.error("No job description provided (either --job-file or --job-text is required).")
        return None

# --- Output Saving ---
def save_warnings(warnings: List[str], output_dir: str = ".", filename: str = "warnings.log") -> None:
    """Saves warnings to a log file."""
    filepath = os.path.join(output_dir, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for warning in warnings:
                f.write(f"WARNING: {warning}\n")
        logger.info(f"Warnings saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving warnings to {filepath}: {e}")

def save_substitutions(substitutions: Dict[str, str], output_dir: str = ".", filename: str = "substitutions.json") -> None:
    """Saves technology substitutions to a JSON file."""
    filepath = os.path.join(output_dir, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(substitutions, f, indent=2)
        logger.info(f"Substitutions saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving substitutions to {filepath}: {e}")

def format_experience_output(experience_dict: Dict) -> str:
    """Formats the structured experience dictionary back into a string similar to YAML."""
    lines = []
    for job, content in experience_dict.items():
        lines.append(f"{job}:")
        if isinstance(content, list): # Bullet points directly under job
            for bullet in content:
                lines.append(f"  - {bullet}")
        elif isinstance(content, dict): # Projects under job
            for project, bullets in content.items():
                lines.append(f"  {project}:")
                if isinstance(bullets, list):
                    for bullet in bullets:
                        lines.append(f"    - {bullet}")
        lines.append("") # Add blank line between jobs
    return "\n".join(lines)


# --- Skill File Parsing ---
def parse_skills_file(filepath: str) -> List[str]:
    """
    Parses the cv/skills.txt file, extracting unique skills.
    Ignores headers and handles comma-separated skills on lines starting with '•'.
    """
    unique_skills: Set[str] = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                # Check if line starts with bullet point (allowing for potential whitespace)
                if line.startswith('•'):
                    # Remove bullet point and leading space
                    line_content = line.lstrip('•').strip()
                    # Split at the first colon to separate header (ignored) from skills
                    parts = line_content.split(':', 1)
                    if len(parts) == 2:
                        skills_str = parts[1] # Take the part after the colon
                        # Split skills by comma and clean them
                        skills = [skill.strip() for skill in skills_str.split(',') if skill.strip()]
                        unique_skills.update(skills)
                    elif len(parts) == 1 and parts[0]: # Handle lines with only skills after bullet
                         skills = [skill.strip() for skill in parts[0].split(',') if skill.strip()]
                         unique_skills.update(skills)

    except FileNotFoundError:
        logger.error(f"Skills file not found: {filepath}")
        return []
    except Exception as e:
        logger.error(f"Error parsing skills file {filepath}: {e}", exc_info=True)
        return []

    logger.info(f"Parsed {len(unique_skills)} unique skills from {filepath}")
    logger.info(f"Parsed {len(unique_skills)} unique skills from {filepath}")
    # Normalize skills (lowercase) before returning
    normalized_skills = {skill.lower() for skill in unique_skills}
    return sorted(list(normalized_skills))


# --- Change Detection ---
def _calculate_file_hash(filepath: str) -> Optional[str]:
    """Calculates the SHA256 hash of a file."""
    hasher = hashlib.sha256()
    try:
        with open(filepath, 'rb') as file:
            while True:
                chunk = file.read(4096) # Read in chunks
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        logger.error(f"File not found for hashing: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error calculating hash for {filepath}: {e}", exc_info=True)
        return None

def check_file_changed(filepath_to_check: str, hash_filepath: str) -> bool:
    """Checks if a file has changed since the last stored hash."""
    current_hash = _calculate_file_hash(filepath_to_check)
    if current_hash is None:
        return True # Assume changed if current file can't be hashed

    stored_hash = None
    try:
        if os.path.exists(hash_filepath):
            with open(hash_filepath, 'r') as f:
                stored_hash = f.read().strip()
    except Exception as e:
        logger.warning(f"Could not read stored hash file {hash_filepath}: {e}. Assuming changed.")
        return True

    if current_hash != stored_hash:
        logger.info(f"Change detected in {os.path.basename(filepath_to_check)} (hash mismatch).")
        return True
    else:
        logger.info(f"No change detected in {os.path.basename(filepath_to_check)}.")
        return False

def store_file_hash(filepath_to_hash: str, hash_filepath: str) -> None:
    """Calculates and stores the hash of the specified file."""
    current_hash = _calculate_file_hash(filepath_to_hash)
    if current_hash:
        try:
            Path(hash_filepath).parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            with open(hash_filepath, 'w') as f:
                f.write(current_hash)
            logger.info(f"Stored new hash for {os.path.basename(filepath_to_hash)} in {os.path.basename(hash_filepath)}.")
        except Exception as e:
            logger.error(f"Error storing hash to {hash_filepath}: {e}", exc_info=True)
