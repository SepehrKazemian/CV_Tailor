import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

# Specify the model name - consider making this configurable
DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2' # A popular, efficient choice

class SkillEmbedder:
    """Handles generation of embeddings for skills."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        """
        Initializes the SentenceTransformer model.

        Args:
            model_name: The name of the sentence-transformer model to load.
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Sentence transformer model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model '{self.model_name}': {e}", exc_info=True)
            # Depending on requirements, could raise error or allow graceful failure

    def get_embeddings(self, skills: List[str]) -> Optional[List[List[float]]]:
        """
        Generates embeddings for a list of skill strings.

        Args:
            skills: A list of skill strings.

        Returns:
            A list of embeddings (each embedding is a list of floats),
            or None if the model failed to load or an error occurred.
        """
        if not self.model:
            logger.error("Embedding model not loaded. Cannot generate embeddings.")
            return None
        if not skills:
            logger.warning("Received empty list of skills for embedding.")
            return []

        try:
            logger.debug(f"Generating embeddings for {len(skills)} skills...")
            # The encode method returns numpy arrays, convert to list of lists for JSON/Neo4j compatibility
            embeddings_np: np.ndarray = self.model.encode(skills, show_progress_bar=False)
            embeddings_list = embeddings_np.tolist()
            logger.debug("Embeddings generated successfully.")
            return embeddings_list
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}", exc_info=True)
            return None

# --- Test Execution ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    sample_skills = ["Python", "TensorFlow", "AWS S3", "Langchain", "Docker", "SQL Databases"]

    logger.info("--- Running SkillEmbedder Test ---")
    try:
        embedder = SkillEmbedder()
        if embedder.model:
            embeddings = embedder.get_embeddings(sample_skills)
            if embeddings:
                print(f"\nSuccessfully generated {len(embeddings)} embeddings.")
                print(f"Embedding dimension: {len(embeddings[0])}")
                # print("Sample embedding (first 5 dims of first skill):")
                # print(embeddings[0][:5])
            else:
                print("\nFailed to generate embeddings. Check logs.")
        else:
            print("\nFailed to load embedding model. Cannot run test.")

        logger.info("--- SkillEmbedder Test Finished ---")

    except Exception as e:
        logger.error(f"An unexpected error occurred during the test: {e}", exc_info=True)
