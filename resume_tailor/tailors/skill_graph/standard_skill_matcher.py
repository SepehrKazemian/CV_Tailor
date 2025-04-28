import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Optional
import re


def normalize_text(text: str) -> str:
    """Removes non-alphanumeric characters and lowercases the text."""
    return re.sub(r'[^a-zA-Z0-9]', '', text).lower()

class StandardSkillMatcher:
    def __init__(self, embedding_model, storage_path: str, similarity_threshold=0.9):
        """
        Args:
            embedding_model: sentence-transformer model to embed skills.
            storage_path: path prefix (without extension) to save/load.
            similarity_threshold: minimum cosine similarity to accept match.
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.storage_path = storage_path

        self.standard_skills: List[str] = []          # canonical standardized skills
        self.standard_embeddings: Optional[np.ndarray] = None
        self.raw_to_standard_map: Dict[str, str] = {} # raw (alias) -> canonical skill

        self._load_resources()

    def _load_resources(self):
        """Load standard skills and embeddings if saved."""
        try:
            skills_path = self.storage_path / "standard_skills.json"
            embeddings_path = self.storage_path / "standard_embeddings.npz"

            if os.path.exists(skills_path) and os.path.exists(embeddings_path):
                with open(skills_path, "r", encoding="utf-8") as f:
                    self.raw_to_standard_map = json.load(f)
                data = np.load(embeddings_path, allow_pickle=True)
                self.standard_skills = data["skills"].tolist()
                self.standard_embeddings = data["embeddings"]

                print(f"Loaded {len(self.standard_skills)} standardized skills.")
            else:
                print("No saved standard skills or embeddings. Fresh start.")
        except Exception as e:
            print(f"Error loading resources: {e}")

    def save_resources(self):
        """Save standard skills and embeddings."""
        try:
            skills_path = self.storage_path / "standard_skills.json"
            embeddings_path = self.storage_path / "standard_embeddings.npz"

            with open(skills_path, "w", encoding="utf-8") as f:
                json.dump(self.raw_to_standard_map, f, ensure_ascii=False, indent=2)
            np.savez(embeddings_path, skills=self.standard_skills, embeddings=self.standard_embeddings)

            print(f"Saved {len(self.standard_skills)} standardized skills and embeddings.")
        except Exception as e:
            print(f"Error saving resources: {e}")

    def update_standards(self, new_raw_to_standard_map: Dict[str, str]):
        """
        Update raw-to-standard mapping and only embed new skills correctly.
        Both raw and standardized skills are normalized before embedding.
        """
        self.raw_to_standard_map.update(new_raw_to_standard_map)

        all_new_skills = set(new_raw_to_standard_map.keys()) | set(new_raw_to_standard_map.values())

        # Normalize for embedding
        normalized_to_original = {}
        for skill in all_new_skills:
            normalized = normalize_text(skill)
            normalized_to_original[normalized] = skill  # Save mapping from normalized to original

        already_embedded_norms = set(normalize_text(s) for s in self.standard_skills)
        skills_to_embed_normed = sorted(set(normalized_to_original.keys()) - already_embedded_norms)

        if not skills_to_embed_normed:
            print("No new skills to embed.")
            return

        print(f"Embedding {len(skills_to_embed_normed)} new normalized skills...")

        # 🚨 Embed the normalized strings, NOT the original messy text
        new_embeddings = self.embedding_model.encode(skills_to_embed_normed, normalize_embeddings=True)

        # Extend storage: store the *original human-readable* form, but match normalized embeddings
        for norm_skill in skills_to_embed_normed:
            self.standard_skills.append(normalized_to_original[norm_skill])

        if self.standard_embeddings is None:
            self.standard_embeddings = new_embeddings
        else:
            self.standard_embeddings = np.vstack([self.standard_embeddings, new_embeddings])

        self.save_resources()

    def match_skills(self, skills: List[str]) -> Dict[str, List[str]]:
        """
        Matches input skills to all standardized skills above threshold.
        
        Args:
            skills: list of raw skill strings.

        Returns:
            dict: {input skill: list of matching standardized skills} (empty list if no match)
        """
        if not skills:
            return {}

        if not self.standard_skills or self.standard_embeddings is None:
            print("No standard skills loaded. Returning empty matches.")
            return {skill: [] for skill in skills}

        # 🚨 Normalize incoming skills before embedding
        normalized_skills = [normalize_text(skill) for skill in skills]

        print(f"Embedding {len(skills)} normalized skills for matching...")
        skill_embeddings = self.embedding_model.encode(normalized_skills, normalize_embeddings=True)

        cosine_scores = util.cos_sim(skill_embeddings, self.standard_embeddings).numpy()

        matches = {}
        for i, skill in enumerate(skills):
            skill_scores = cosine_scores[i]
            high_score_indices = np.where(skill_scores >= self.similarity_threshold)[0]

            matched_standards = []
            if len(high_score_indices) > 0:
                for idx in high_score_indices:
                    matched_raw = self.standard_skills[idx]  # Human-readable saved form
                    standard_skill = self.raw_to_standard_map.get(matched_raw, matched_raw)
                    matched_standards.append(standard_skill)

                matched_standards = list(sorted(set(matched_standards)))

            matches[skill] = matched_standards

        return matches


