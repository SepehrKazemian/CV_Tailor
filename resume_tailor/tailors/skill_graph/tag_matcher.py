import logging
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, util
from resume_tailor.tailors.skills.tags.load_tags_stackoverflow import parse_stackoverflow_tags
from resume_tailor.tailors.skills.tags.tag_embedding_loader import compute_tag_embeddings

logger = logging.getLogger(__name__)

# --- Configuration ---
STACKOVERFLOW_EMBEDDING_MODEL = 'intfloat/e5-base-v2'
SIMILARITY_THRESHOLD = 0.95

class StackOverflowTagger:
    """Matches skills against Stack Overflow tags using embeddings."""

    def __init__(self, model_name: str = STACKOVERFLOW_EMBEDDING_MODEL):
        self.model_name = model_name
        self.model: SentenceTransformer | None = None
        self.tag_names: List[str] | None = None
        self.tag_embeddings: np.ndarray | None = None
        self._load_resources()

    def _load_resources(self):
        """Loads the embedding model and tag data."""
        try:
            logger.info(f"Loading Stack Overflow tag data...")
            tag_set, tag_counts = parse_stackoverflow_tags() # Assuming this returns dict {tag: count}
            if not tag_counts:
                raise ValueError("Failed to parse Stack Overflow tags.")

            logger.info(f"Loading/Computing Stack Overflow tag embeddings for model '{self.model_name}'...")
            # compute_tag_embeddings returns tuple: (tag_names, tag_vecs, log_weights, weighted_centroid)
            tag_data = compute_tag_embeddings(tag_counts, model_name=self.model_name)
            if not tag_data or len(tag_data) < 2:
                 raise ValueError("Failed to compute/load tag embeddings.")

            self.tag_names = list(tag_data)[0]
            self.tag_embeddings = list(tag_data)[1] # This should be the numpy array

            # Load the specific model used for tag embeddings if needed for query encoding
            # Note: compute_tag_embeddings already loads it, but we might need it here too
            # Re-using the model from compute_tag_embeddings might be more efficient if possible
            # For simplicity now, load it again if needed.
            # self.model = SentenceTransformer(self.model_name)

            logger.info(f"Successfully loaded {len(self.tag_names)} Stack Overflow tags and embeddings.")

        except Exception as e:
            logger.error(f"Failed to initialize StackOverflowTagger resources: {e}", exc_info=True)
            self.tag_names = None
            self.tag_embeddings = None
            # self.model = None # Ensure model is None if loading failed

    def get_matches(self, skills: List[str], skill_embeddings: np.ndarray) -> Dict[str, List[str]]:
        """
        Finds high-confidence Stack Overflow tag matches for given skills.

        Args:
            skills: List of normalized skill names.
            skill_embeddings: Numpy array of corresponding skill embeddings (must match model used for tags).

        Returns:
            Dictionary mapping each input skill to a list of matching tag names (similarity >= threshold).
        """
        matches: Dict[str, List[str]] = {skill: [] for skill in skills}
        if self.tag_names is None or self.tag_embeddings is None or skill_embeddings is None or len(skills) != len(skill_embeddings):
            logger.error("Tagger not initialized or input mismatch. Cannot find matches.")
            return matches
        if len(skills) == 0:
            return matches

        logger.info(f"Finding Stack Overflow tag matches for {len(skills)} skills (Threshold: {SIMILARITY_THRESHOLD})...")
        try:
            # Compute cosine similarities between skill embeddings and all tag embeddings
            # util.cos_sim returns a tensor, convert to numpy
            cosine_scores = util.cos_sim(skill_embeddings, self.tag_embeddings).numpy()

            for i, skill in enumerate(skills):
                # Get scores for the i-th skill against all tags
                skill_scores = cosine_scores[i]
                # Find indices where score meets the threshold
                high_score_indices = np.where(skill_scores >= SIMILARITY_THRESHOLD)[0]

                if len(high_score_indices) > 0:
                    matched_tags = [self.tag_names[idx] for idx in high_score_indices]
                    # Optional: Sort matches by score? For now, just return all above threshold.
                    # sorted_indices = high_score_indices[np.argsort(skill_scores[high_score_indices])[::-1]]
                    # matched_tags = [self.tag_names[idx] for idx in sorted_indices]
                    matches[skill] = matched_tags
                    logger.debug(f"Skill '{skill}' matched tags: {matched_tags}")

        except Exception as e:
            logger.error(f"Error during tag matching: {e}", exc_info=True)
            # Return potentially partial matches found so far
            return matches

        return matches

# --- Test Execution ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("--- Running StackOverflowTagger Test ---")

    # Need an embedding model instance for testing
    test_model_name = STACKOVERFLOW_EMBEDDING_MODEL
    try:
        logger.info(f"Loading model '{test_model_name}' for test embedding generation...")
        test_embedder = SentenceTransformer(test_model_name)

        sample_skills = ["python", "tensorflow", "react", "aws s3", "kubernetes", "vector database"]
        logger.info(f"Generating embeddings for test skills: {sample_skills}")
        sample_embeddings = test_embedder.encode(sample_skills, normalize_embeddings=True)

        tagger = StackOverflowTagger(model_name=test_model_name)
        if tagger.tag_names and tagger.tag_embeddings is not None:
            skill_to_tags_map = tagger.get_matches(sample_skills, sample_embeddings)

            print("\n--- Skill to Stack Overflow Tag Matches (>= 0.95) ---")
            for skill, tags in skill_to_tags_map.items():
                print(f"'{skill}': {tags if tags else 'No high-confidence matches'}")
        else:
            print("\nFailed to initialize tagger or load resources.")

    except Exception as e:
        logger.error(f"An unexpected error occurred during the test: {e}", exc_info=True)

    logger.info("--- StackOverflowTagger Test Finished ---")
