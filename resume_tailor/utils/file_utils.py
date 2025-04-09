import os
import json
import logging
from typing import Dict, List, Optional
import yaml # Import YAML

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
