# resume_tailor/tailors/skill_graph/skill_preprocessing.py
import logging
import re
from typing import List, Dict, Tuple, Set, Optional
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer
import numpy as np
from resume_tailor.tailors.skill_graph.skill_graph_llms import standardize_skills_llm
from resume_tailor.tailors.skill_graph.standard_skill_matcher import StandardSkillMatcher
import resume_tailor.tailors.skill_graph.skill_graph_config as sgc

# Corrected absolute imports
from resume_tailor.tailors.skill_graph.tag_matcher import StackOverflowTagger
# from resume_tailor.tailors.skills.tags.tag_embedding_loader import compute_tag_embeddings # Placeholder - Absolute path if uncommented
# from resume_tailor.tailors.skills.tags.load_tags_stackoverflow import parse_stackoverflow_tags # Placeholder - Absolute path if uncommented

# Corrected absolute import
from resume_tailor.utils.llm_provider import get_llm
# Imports for run_llm_chain and extract_json_block are removed as they are not used
import json
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

# --- Normalization Helper ---
def normalize_fuzzy(text: str) -> str:
    """Lowercase and remove non-alphanumeric characters for fuzzy matching."""
    return re.sub(r'[^a-z0-9]', '', text.lower())

# --- Pydantic Model for LLM Standardization ---
class StandardizedSkill(BaseModel):
    standardized_name: str = Field(..., description="The single best canonical string for the skill.")

class StandardizationDict(BaseModel):
    standardizations: Dict[str, StandardizedSkill] = Field(
        ...,
        description="Dictionary where key is the raw skill and value contains the standardized name."
    )


class SkillPreprocessor:
    """Handles normalization and standardization of skill lists."""

    def __init__(self, llm_provider: str = 'anthropic', api_key: Optional[str] = None):
        """
        Initializes the preprocessor.
        Args:
            llm_provider: The provider for the standardization LLM ('openai', 'google', 'anthropic').
            api_key: Optional API key if needed by the provider.
        """
        # TODO: Load StackOverflow tag data (tag_names, tag_vecs) here or ensure it's passed
        # For now, assume StackOverflowTagger handles its own data loading if needed.
        self.tagger = StackOverflowTagger()
        self.embedding_model = SentenceTransformer("intfloat/e5-base-v2") # Or use embedder from graph_builder

        # Initialize LLM for standardization - Get Langchain wrapper AND raw client
        self.std_llm = None # Langchain wrapper (might not be needed if only using raw client)
        self.raw_client = None # Raw Anthropic client
        self.std_model_name = None
        self.standard_matcher = StandardSkillMatcher(self.embedding_model, sgc.dataset_folder)
        try:
            # Get all return values from get_llm
            self.std_model_name, self.std_llm, self.raw_client = get_llm(
                provider=llm_provider,
                api_key=api_key,
                return_raw=True # Ensure raw client is requested
            )
            logger.info(f"Initialized Standardization LLM provider '{llm_provider}' with model: {self.std_model_name}")
            if not self.raw_client and llm_provider in ['anthropic', 'claude']:
                 logger.warning("Raw Anthropic client could not be initialized. Standardization LLM call will fail.")
            # Log if Langchain wrapper failed, though we might not use it
            if not self.std_llm:
                 logger.warning("Langchain wrapper for standardization LLM could not be initialized.")

        except Exception as e:
            logger.error(f"Failed to initialize LLM provider for standardization: {e}", exc_info=True)
            self.std_llm = None
            self.raw_client = None


    def _embed_skills(self, skills: List[str]) -> Tuple[Dict[str, str], Optional[np.ndarray]]:
        """Embeds skills, handling potential replacements for embedding."""
        # Replace problematic characters for embedding model if necessary
        # Example: replace '/' and ' ' with '-'
        embed_friendly_skills = {s: s.replace("/", "-").replace(" ", "-") for s in skills}
        skill_keys_for_embedding = list(embed_friendly_skills.values())

        if not skill_keys_for_embedding:
            return {}, None

        try:
            logger.info(f"Embedding {len(skill_keys_for_embedding)} skills for standardization...")
            skill_vecs = self.embedding_model.encode(skill_keys_for_embedding, normalize_embeddings=True)
            logger.info("Skill embedding complete.")
            return embed_friendly_skills, skill_vecs
        except Exception as e:
            logger.error(f"Failed to embed skills: {e}", exc_info=True)
            return embed_friendly_skills, None

    def _get_matches_with_fuzzy(self, matches: Dict[str, List[str]]) -> Dict[str, Dict]:
        """
        Given matches {raw_skill: [matched_standard_skills]}, 
        returns {raw_skill: {'original': [matches], 'fuzzy_scores': [scores]}}.
        """
        matches_ratio = {}  # 🛠 You missed this line

        for raw_skill, matched_standards in matches.items():
            if not matched_standards:
                matches_ratio[raw_skill] = {
                    "original": [],
                    "fuzzy_scores": []
                }
                continue

            norm_raw = normalize_text(raw_skill)

            fuzzy_scores = []
            for standard_skill in matched_standards:
                norm_standard = normalize_text(standard_skill)
                score = fuzz.ratio(norm_raw, norm_standard)
                fuzzy_scores.append(score)

            matches_ratio[raw_skill] = {
                "original": matched_standards,
                "fuzzy_scores": fuzzy_scores
            }

        logger.info("Fuzzy matching complete.")
        return matches_ratio


    def _categorize_matches(self, matches_ratio: Dict[str, Dict], original_skill_map: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, List[str]]]:
        """Categorizes matches into perfect fuzzy, other matches, and no matches."""
        perfect_matches_map = {} # {original_skill: standardized_name}
        semi_match = {} # {original_skill: [so_matches]}
        no_match = {} # {original_skill: []}

        # Map embed_key back to original skill name
        reverse_skill_map = {v: k for k, v in original_skill_map.items()}

        logger.info("Categorizing SO matches...")
        for embed_key, value in matches_ratio.items():
            original_skill = reverse_skill_map.get(embed_key)
            if not original_skill:
                logger.warning(f"Could not map embed key '{embed_key}' back to an original skill. Skipping.")
                continue

            if "fuzzy_scores" in value and 100 in value["fuzzy_scores"]:
                index = value["fuzzy_scores"].index(100)
                # Perfect fuzzy match means the SO tag is essentially identical after normalization
                perfect_match_tag = value["original"][index]
                perfect_matches_map[original_skill] = perfect_match_tag
                logger.debug(f"Perfect fuzzy match for '{original_skill}': '{perfect_match_tag}'")
            elif "original" in value and value["original"]:
                # Has SO matches, but none are perfect fuzzy matches - needs LLM review
                semi_match[original_skill] = value["original"]
                logger.debug(f"Needs LLM standardization for '{original_skill}': SO Matches {value['original']}")
            else:
                # No SO matches found at all - needs LLM review (with empty context)
                no_match[original_skill] = []
                logger.debug(f"No SO matches found for '{original_skill}'. Needs LLM standardization.")

        # Combine skills needing LLM review
        logger.info(f"Categorized matches: {len(perfect_matches_map)} perfect, {len(no_match) + len(semi_match)} need LLM.")

        # Return the map of skills needing LLM review separately from those with no matches initially
        # This might be useful if the LLM prompt needs to differentiate
        return perfect_matches_map, semi_match, no_match


    def run(self, raw_skills: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """
        Runs the full preprocessing pipeline.

        Args:
            raw_skills: A list of skill strings to process.

        Returns:
            A tuple containing:
            - A list of unique, standardized skill names (excluding filtered ones).
            - A dictionary mapping original raw skills to their standardized name
              (value is None if filtered out e.g. soft skills or empty string from LLM).
        """
        if not raw_skills:
            return [], {}

        # Ensure input skills are strings and stripped
        unique_raw_skills = sorted(list(set(str(s).strip() for s in raw_skills if s and str(s).strip())))
        if not unique_raw_skills:
            return [], {}

        logger.info(f"Starting preprocessing for {len(unique_raw_skills)} unique raw skills.")

        matches = self.standard_matcher.match_skills(unique_raw_skills)

        # 2. Get SO Matches + Fuzzy Scores
        # Use the keys from original_skill_map (embed-friendly names) for matching
        matches_ratio = self._get_matches_with_fuzzy(matches)
        # matches_ratio: {embed_friendly_skill: {"original": [so_tag], "fuzzy_scores": [score]}}

        # 3. Categorize Matches
        perfect_matches_map, semi_match, no_match = self._categorize_matches(matches_ratio, original_skill_map)
        # perfect_matches_map: {original_skill: standardized_name}
        # needs_llm_standardization: {original_skill: [so_matches]}

        # 4. Standardize remaining with LLM
        if semi_match:
            llm_standardized_semi_match_map = standardize_skills_llm(semi_match, raw_client)
        else:
            llm_standardized_semi_match_map = {}
        if no_match:
            llm_standardized_no_match_map = standardize_skills_llm(no_match, raw_client)
        else:
            llm_standardized_no_match_map = {}
            
        # llm_standardized_map: {original_skill: standardized_name} (standardized_name can be "" here)

        # 5. Combine results and create final mapping
        # This map tracks the final standardized name for *every* unique raw skill input
        final_standardized_map_nullable: Dict[str, Optional[str]] = {}
        final_standardized_skills_set: Set[str] = set() # Only stores non-empty, standardized names

        final_standardized_skills_set, final_standardized_map_nullable, soft_skills_list = self._merge_standardized_map_with_soft_skills(
            perfect_matches_map,
            llm_standardized_semi_match_map,
            llm_standardized_no_match_map
        )
        
        self.standard_matcher.update_standards(final_standardized_map_nullable)
        
        return final_standardized_skills_set, final_standardized_map_nullable, soft_skills_list

        # self._ensure_completeness(unique_raw_skills, final_standardized_map_nullable)
        
    def _merge_standardized_map_with_soft_skills(
            self,
            perfect_matches_map: dict,
            llm_standardized_semi_match_map: dict,
            llm_standardized_no_match_map: dict,
        ) -> tuple[set, dict, set]:
        final_map = {}
        final_set = set()
        soft_skills = set()

        # Process perfect matches
        for original, standardized in perfect_matches_map.items():
            final_map[original] = standardized
            final_set.add(standardized)

        # Process semi matches
        for original, standardized in llm_standardized_semi_match_map.items():
            if len(standardized) == 0:
                soft_skills.add(original)
            else:
                final_map[original] = standardized
                final_set.add(standardized)

        # Process no matches
        for original, standardized in llm_standardized_no_match_map.items():
            if len(standardized) == 0:
                soft_skills.add(original)
            else:
                final_map[original] = standardized
                final_set.add(standardized)

        return final_set, final_map, soft_skills


    def _ensure_completeness(
        self, raw_skills: list[str], final_map: dict
    ) -> None:
        for original in raw_skills:
            if original not in final_map:
                logger.warning(f"Original skill '{original}' missing from final map. Marking as filtered.")
                final_map[original] = None


    def _finalize_standardization_output(
        self, final_set: set, final_map: dict
    ) -> tuple[list[str], dict]:
        final_list = sorted(list(final_set))
        logger.info(f"Preprocessing complete. Produced {len(final_list)} standardized skills.")
        logger.debug(f"Final Standardized List: {final_list}")
        # logger.debug(f"Original to Standardized Map (Nullable): {final_map}")
        return final_list, final_map