# resume_tailor/utils/check_changes_hash.py
import logging
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)

def calculate_file_hash(filepath: Path) -> str:
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

def check_file_changed(filepath: Path, hash_filepath: Path) -> bool:
    """
    Checks if the file content has changed since the last stored hash.

    Args:
        filepath: Path to the file to check.
        hash_filepath: Path to the file storing the hash.

    Returns:
        True if the file has changed or if errors occurred, False otherwise.
    """
    if not filepath.exists():
        logger.warning(f"File to check does not exist: {filepath}")
        # Consider it changed if file is gone but hash exists (to clear hash)
        return hash_filepath.exists()

    current_hash = calculate_file_hash(filepath)
    if not current_hash: return True # Treat hash error as change

    stored_hash = ""
    if hash_filepath.exists():
        try:
            with open(hash_filepath, 'r') as f:
                stored_hash = f.read().strip()
        except Exception as e:
            logger.error(f"Error reading hash file {hash_filepath}: {e}", exc_info=True)
            return True # Treat read error as change

    if current_hash != stored_hash:
        logger.info(f"Change detected in {filepath} (hash mismatch).")
        try:
            hash_filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(hash_filepath, 'w') as f:
                f.write(current_hash)
            logger.info(f"Updated hash stored in {hash_filepath}")
        except Exception as e:
             logger.error(f"Error writing hash file {hash_filepath}: {e}", exc_info=True)
        return True
    else:
        logger.debug(f"No changes detected in {filepath} based on hash.")
        return False

def save_file_hash(filepath: Path, hash_filepath: Path) -> bool:
    """
    Calculates and saves the SHA256 hash of a file to a separate hash file.

    Args:
        filepath: Path to the file to hash.
        hash_filepath: Path where the hash will be saved.

    Returns:
        True if saved successfully, False otherwise.
    """
    if not filepath.exists():
        logger.warning(f"Cannot save hash. File does not exist: {filepath}")
        return False

    file_hash = calculate_file_hash(filepath)
    if not file_hash:
        logger.error(f"Failed to calculate hash for {filepath}")
        return False

    try:
        hash_filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(hash_filepath, 'w') as f:
            f.write(file_hash)
        logger.info(f"Saved file hash to {hash_filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving hash to {hash_filepath}: {e}", exc_info=True)
        return False
