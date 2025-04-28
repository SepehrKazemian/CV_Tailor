import logging
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from resume_tailor.tailors.skill_graph.graph_connector import Neo4jConnector
from resume_tailor.tailors.skill_graph.extractor import SkillExtractor
from resume_tailor.tailors.skill_graph.embedder import SkillEmbedder
# Import tag matcher and other necessary components
from resume_tailor.tailors.skill_graph.tag_matcher import StackOverflowTagger
# SkillPreprocessor is now used by CandidateSkillProcessor, no longer needed here directly
# from .skill_preprocessing import SkillPreprocessor
from resume_tailor.tailors.skill_graph.candidate_skills import CandidateSkillProcessor # Import the new processor
from resume_tailor.tailors.skill_graph.schema import (
    L1L2ValidationResult, L3DeterminationResult, L2NameDeterminationResult, GeneratedL2Categories
)
from resume_tailor.utils.llm_provider import get_llm
from resume_tailor.utils.llm_utils import run_llm_chain
from resume_tailor.tailors.tailor_utils import extract_json_block
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pydantic import ValidationError
import re
from pathlib import Path # Added import

logger = logging.getLogger(__name__)

# --- Constants and Initial Data ---
INITIAL_L3_DOMAINS = [
    "Software Engineering", "Data Science", "Machine Learning", "Deep Learning", "LLM",
    "NLP", "Computer Vision", "Cloud Computing", "DevOps", "MLOps", "Data Engineering",
    "Cybersecurity", "Web Development", "Mobile Development", "Database Management",
    "Business Intelligence", "Project Management", "Cognitive Skills", "Leadership",
    "Communication", "Computer Science", "Information Retrieval"
]
INITIAL_L2_CATEGORIES = {
    "Programming Languages": "Software Engineering", "Web Frameworks": "Software Engineering",
    "API Development": "Software Engineering", "Software Architecture": "Software Engineering",
    "Testing Frameworks": "Software Engineering", "Data Analysis Libraries": "Data Science",
    "Visualization Tools": "Data Science", "Statistical Modeling": "Data Science",
    "Experimentation Platforms": "Data Science", "ML Frameworks": "Machine Learning",
    "Classical ML Algorithms": "Machine Learning", "Feature Engineering": "Machine Learning",
    "Model Evaluation": "Machine Learning", "DL Frameworks": "Deep Learning",
    "Neural Network Architectures": "Deep Learning", "Sequence Models": "Deep Learning",
    "LLM Frameworks": "LLM", "LLM Models": "LLM", "Vector Databases": "LLM",
    "Prompt Engineering": "LLM", "Agent Frameworks": "LLM", "NLP Libraries": "NLP",
    "Text Processing Techniques": "NLP", "Information Extraction": "NLP",
    "CV Libraries": "Computer Vision", "Image Processing": "Computer Vision",
    "Object Detection/Recognition": "Computer Vision", "Cloud Platforms": "Cloud Computing",
    "Serverless Computing": "Cloud Computing", "Infrastructure as Code": "Cloud Computing",
    "Container Orchestration": "Cloud Computing", "CI/CD Tools": "DevOps",
    "Version Control Systems": "DevOps", "Configuration Management": "DevOps",
    "Experiment Tracking": "MLOps", "Model Serving Platforms": "MLOps",
    "Monitoring & Observability Tools": "MLOps", "Data Versioning": "MLOps",
    "Data Warehousing": "Data Engineering", "ETL/ELT Tools": "Data Engineering",
    "Streaming Platforms": "Data Engineering", "Data Orchestration": "Data Engineering",
    "Security Frameworks": "Cybersecurity", "Threat Detection": "Cybersecurity",
    "Compliance Standards": "Cybersecurity", "Frontend Frameworks": "Web Development",
    "Backend Frameworks": "Web Development", "Mobile Platforms": "Mobile Development",
    "Cross-Platform Frameworks": "Mobile Development", "SQL Databases": "Database Management",
    "NoSQL Databases": "Database Management", "BI Platforms": "Business Intelligence",
    "Data Visualization": "Business Intelligence", "Agile Methodologies": "Project Management",
    "Project Tracking Tools": "Project Management",
    "Workflow Orchestration Platforms": "Data Engineering", "Platforms": "Cloud Computing",
    "ML Platforms": "Machine Learning", "Cognitive Skills": "Cognitive Skills",
    "Tools": "DevOps", "Serverless Platforms": "Cloud Computing", "Monitoring Tools": "Cloud Computing",
    "Techniques": "Data Science", "Data Warehouses": "Data Engineering", "AI Services": "Computer Vision",
    "Thinking Skills": "Cognitive Skills", "Utilities": "DevOps", "Interpersonal Skills": "Cognitive Skills",
    "Data Generation Techniques": "Data Engineering", "Orchestration Frameworks": "Data Engineering",
    "Data Transformation Tools": "Data Engineering", "NLP Tasks": "NLP", "Search Engines": "Data Engineering",
    "Access Management": "Cybersecurity", "Business Concepts": "Business Intelligence", "BI Tools": "Business Intelligence",
    "Soft Skills": "Project Management", "Libraries": "Software Engineering", "Statistical Concepts": "Data Science",
    "Statistical Methods": "Data Science", "ML Libraries": "Machine Learning", "ML Explainability Libraries": "Machine Learning",
    "Data Processing Frameworks": "Data Engineering", "Thinking Approaches": "Cognitive Skills",
    "Data Visualization Tools": "Business Intelligence", "Teamwork Skills": "Leadership",
    "OCR Libraries": "Computer Vision", "Architectures": "NLP", "Communication Skills": "Communication",
    "Database Systems": "Database Management", "Concepts": "Software Engineering",
    "Validation Techniques": "Machine Learning", "LLM Libraries": "LLM",
    "ML Platforms": "MLOps",
    "Learning Strategies": "Cognitive Skills",
    "Memory Techniques": "Cognitive Skills", "Problem Solving Approaches": "Cognitive Skills",
    "Creativity Methods": "Cognitive Skills", "Decision Making Models": "Cognitive Skills",
    "Leadership Styles": "Leadership", "Management Techniques": "Leadership",
    "Motivation Theories": "Leadership", "Team Building": "Leadership",
    "Organizational Culture": "Leadership", "Decision Making": "Leadership",
    "People Development": "Leadership", "Musical Instruments": "Music",
    "Music Genres": "Music", "Music Theory Concepts": "Music",
    "Audio Production Tools": "Music", "Music Notation Standards": "Music",
    "Vocal Techniques": "Music", "Learning Abilities": "Cognitive Skills",
    "Communication Protocols": "Communication", "Messaging Platforms": "Communication",
    "Collaboration Tools": "Communication", "Networking Standards": "Communication",
    "Telecommunication Technologies": "Communication", "Academic Disciplines": "Computer Science",
    "Data Structures": "Computer Science", "Algorithms": "Computer Science",
    "Operating Systems": "Computer Science", "Computer Architecture": "Computer Science",
    "Databases": "Computer Science", "AI Models": "LLM", "Practices": "MLOps",
    "Database Types": "Database Management", "ML Algorithms": "Machine Learning",
    "Registries": "DevOps", "Prompting Techniques": "NLP", "Data Generation": "Data Engineering",
    "Query Expansion Techniques": "Information Retrieval", "NLP APIs": "NLP",
    "Soft Skills": "Soft Skills",
}

# Helper for normalization - Enhanced
def normalize_name(name: str) -> str:
    """Lowercase, remove simple plurals, spaces, and common suffixes for comparison."""
    if not isinstance(name, str): return ""
    name = name.lower().strip()
    suffixes_to_remove = [
        ' platforms', ' libraries', ' frameworks', ' concepts', ' techniques', ' methods',
        ' approaches', ' models', ' systems', ' skills', ' technologies', ' standards',
        ' protocols', ' algorithms', ' services', ' databases', ' engines', ' tools'
    ]
    for suffix in suffixes_to_remove:
        if name.endswith(suffix): name = name[:-len(suffix)]; break
    if name.endswith('es'): name = name[:-2]
    elif name.endswith('s'): name = name[:-1]
    name = re.sub(r'[\s-]+', '', name)
    return name

class SkillGraphBuilder:
    """Builds and updates the skill knowledge graph in Neo4j."""
    SIMILARITY_THRESHOLD = 0.4
    TOP_N_CANDIDATES = 5

    def __init__(self):
        """Initializes the builder with connectors, extractor, embedder, and LLM."""
        self.connector = Neo4jConnector()
        if not self.connector._driver: raise ConnectionError("Failed Neo4j connection.")
        self.extractor = SkillExtractor()
        self.embedder = SkillEmbedder() # Uses default 'all-MiniLM-L6-v2'
        if not self.embedder.model: logger.warning("Embedding model failed to load.")
        # Initialize Stack Overflow Tagger (uses 'intfloat/e5-base-v2' by default)
        self.so_tagger = StackOverflowTagger()
        if not self.so_tagger.tag_names: logger.warning("Stack Overflow Tagger failed to initialize.")

        judge_provider = 'anthropic'
        if 'gpt' in self.extractor.model_name.lower(): judge_provider = 'openai'
        elif 'gemini' in self.extractor.model_name.lower(): judge_provider = 'google'
        elif 'claude' in self.extractor.model_name.lower(): judge_provider = 'anthropic'
        self.judge_model_name, self.judge_llm, _ = get_llm(provider=judge_provider, api_key=None, return_raw=False)
        logger.info(f"Initialized Judge LLM provider '{judge_provider}' with model: {self.judge_model_name}")
        # CandidateSkillProcessor handles its own preprocessor initialization
        # self.preprocessor = SkillPreprocessor(llm_provider=judge_provider)
        self._l2_repr_embedding_cache: Optional[Dict[str, List[float]]] = None
        self._all_l2_names_cache: Optional[Set[str]] = None
        self._all_l3_names_cache: Optional[Set[str]] = None
        # These will be populated by processing methods using CandidateSkillProcessor results
        self.candidate_skills_standardized: Set[str] = set()
        self.jd_skills_standardized: Set[str] = set()
        # The map is now primarily managed by CandidateSkillProcessor, but builder might need access
        self.skill_standardization_map: Dict[str, Optional[str]] = {}
        # File paths and handling helpers are removed from here


    # --- Initialization Methods ---
    def initialize_l3_domains(self, domains: List[str] = INITIAL_L3_DOMAINS):
        """Ensures the initial L3 domain nodes exist."""
        if not self.connector._driver: return
        unique_domains = sorted(list(set(domains + ["Soft Skills"])))
        logger.info(f"Initializing/Verifying {len(unique_domains)} L3 domain nodes...")
        query = "UNWIND $domains as domain_name MERGE (l3:L3 {name: domain_name}) RETURN count(l3) as created_count"
        try:
            self.connector.execute_query(query, parameters={"domains": unique_domains})
            logger.info(f"L3 domain initialization query executed for {len(unique_domains)} domains.")
            self._all_l3_names_cache = None
            self.verify_l3_domains(unique_domains)
        except Exception as e: logger.error(f"Failed to initialize L3 domains: {e}", exc_info=True)

    def verify_l3_domains(self, expected_domains: List[str]):
        """Verify that the expected L3 domains exist."""
        if not self.connector._driver: return
        logger.info("Verifying L3 domain nodes...")
        query = "MATCH (l3:L3) WHERE l3.name IN $expected_domains RETURN l3.name as name"
        try:
            results = self.connector.execute_query(query, parameters={"expected_domains": expected_domains})
            found_domains = {res['name'] for res in results}
            expected_set = set(expected_domains)
            missing = expected_set - found_domains
            if not missing: logger.info(f"Successfully verified all {len(expected_domains)} expected L3 domains exist.")
            else: logger.warning(f"Verification failed! Missing L3 domains: {missing}")
            return found_domains
        except Exception as e: logger.error(f"Error verifying L3 domains: {e}", exc_info=True); return set()

    def initialize_l2_categories(self, categories: Dict[str, str] = INITIAL_L2_CATEGORIES):
        """Ensures the initial L2 category nodes exist and are linked."""
        if not self.connector._driver: return
        categories_with_soft = categories.copy()
        categories_with_soft["Soft Skills"] = "Soft Skills"
        logger.info(f"Initializing/Verifying {len(categories_with_soft)} L2 category nodes and relationships...")
        all_l3s = set(categories_with_soft.values())
        self.initialize_l3_domains(list(all_l3s))
        category_list = [{"l2_name": l2, "l3_name": l3} for l2, l3 in categories_with_soft.items()]
        query = """
        UNWIND $category_list as cat
        MATCH (l3:L3 {name: cat.l3_name})
        MERGE (l2:L2 {name: cat.l2_name})
        MERGE (l2)-[:PART_OF]->(l3)
        RETURN count(l2) as processed_count
        """
        try:
            self.connector.execute_query(query, parameters={"category_list": category_list})
            logger.info(f"L2 category initialization query executed.")
            self._all_l2_names_cache = None; self._l2_repr_embedding_cache = None
        except Exception as e: logger.error(f"Failed to initialize L2 categories: {e}", exc_info=True)

    # --- Caching Methods ---
    def _get_all_l2_names(self) -> set:
        """Fetches and caches all existing L2 category names."""
        if self._all_l2_names_cache is None:
            logger.info("Fetching all L2 category names...")
            query = "MATCH (l2:L2) RETURN l2.name as name"
            results = self.connector.execute_query(query)
            self._all_l2_names_cache = {res['name'] for res in results} if results else set()
            logger.info(f"Cached {len(self._all_l2_names_cache)} L2 names.")
        return self._all_l2_names_cache

    def _get_all_l3_names(self) -> set:
        """Fetches and caches all existing L3 domain names."""
        if self._all_l3_names_cache is None:
            logger.info("Fetching all L3 domain names...")
            query = "MATCH (l3:L3) RETURN l3.name as name"
            results = self.connector.execute_query(query)
            self._all_l3_names_cache = {res['name'] for res in results} if results else set()
            logger.info(f"Cached {len(self._all_l3_names_cache)} L3 names.")
        return self._all_l3_names_cache

    # --- Embedding and Matching Methods ---
    def _calculate_representative_l2_embeddings(self) -> Optional[Dict[str, List[float]]]:
        if not self.connector._driver or not self.embedder.model: return None
        logger.info("Calculating/Fetching representative L2 embeddings...")
        query = "MATCH (l2:L2) OPTIONAL MATCH (l1:L1)-[:BELONGS_TO]->(l2) RETURN l2.name as l2_name, collect(l1.embedding) as l1_embeddings"
        results = self.connector.execute_query(query)
        if not results: logger.warning("No L2 categories found for embedding calculation."); return {}
        l2_repr_embeddings = {}
        l2_names_for_fallback = []
        for record in results:
            l2_name = record['l2_name']
            l1_embeddings = [emb for emb in record.get('l1_embeddings', []) if emb]
            if l1_embeddings:
                try: l2_repr_embeddings[l2_name] = np.mean(np.array(l1_embeddings), axis=0).tolist()
                except Exception as e: logger.error(f"Avg embedding error L2 '{l2_name}': {e}"); l2_names_for_fallback.append(l2_name)
            else: l2_names_for_fallback.append(l2_name)
        if l2_names_for_fallback:
            unique_fallback_names = sorted(list(set(l2_names_for_fallback)))
            logger.info(f"Generating fallback embeddings for {len(unique_fallback_names)} L2 names.")
            fallback_embeddings = self.embedder.get_embeddings(unique_fallback_names)
            if fallback_embeddings and len(fallback_embeddings) == len(unique_fallback_names):
                for name, emb in zip(unique_fallback_names, fallback_embeddings):
                    if name not in l2_repr_embeddings: l2_repr_embeddings[name] = emb
            else: logger.error("Failed fallback L2 embedding generation.")
        self._l2_repr_embedding_cache = l2_repr_embeddings
        logger.info(f"Cached representative embeddings for {len(self._l2_repr_embedding_cache)} L2 categories.")
        return self._l2_repr_embedding_cache

    def _find_candidate_l2_matches(self, l1_skill: str, skill_embedding: List[float]) -> List[Tuple[str, float]]:
        """Finds the top N candidate L2 categories based on similarity, excluding self-matches."""
        # ... (implementation remains the same) ...
        l2_repr_embeddings_dict = self._calculate_representative_l2_embeddings()
        if not l2_repr_embeddings_dict: return []
        valid_l2_embeddings, valid_l2_names = [], []
        normalized_l1 = normalize_name(l1_skill)
        for name, emb in l2_repr_embeddings_dict.items():
            if normalize_name(name) == normalized_l1: logger.debug(f"Pre-filtering self-match: L1 '{l1_skill}' vs L2 '{name}'"); continue
            if emb and isinstance(emb, list) and len(emb) > 0: valid_l2_embeddings.append(emb); valid_l2_names.append(name)
            else: logger.warning(f"Invalid embedding L2 '{name}'. Skipping.")
        if not valid_l2_embeddings: return []
        l2_embeddings_np = np.array(valid_l2_embeddings)
        skill_embedding_np = np.array(skill_embedding).reshape(1, -1)
        try:
            similarities = cosine_similarity(skill_embedding_np, l2_embeddings_np)[0]
            candidate_indices = np.argsort(similarities)[::-1]
            top_candidates = []
            for idx in candidate_indices:
                score = similarities[idx]
                if score >= self.SIMILARITY_THRESHOLD:
                    if len(top_candidates) < self.TOP_N_CANDIDATES: top_candidates.append((valid_l2_names[idx], float(score)))
                else: break
            logger.debug(f"Found {len(top_candidates)} candidates for '{l1_skill}' above threshold {self.SIMILARITY_THRESHOLD}.")
            return top_candidates
        except Exception as e: logger.error(f"Error calculating similarity for '{l1_skill}': {e}", exc_info=True); return []

    # --- Judge LLM and Fallback Methods ---
    # _validate_l1_l2_connections, _generate_and_create_l2s_for_new_l3, _determine_l3_for_orphan, _is_soft_skill, _determine_l2_name_for_orphan, _link_l1_to_l2 remain the same
    def _validate_l1_l2_connections(self, l1_skill: str, candidate_l2s: List[str]) -> Tuple[List[str], Optional[str]]:
        if not candidate_l2s: return [], None
        logger.debug(f"Validating L2 candidates for L1 skill '{l1_skill}': {candidate_l2s}")
        judge_prompt = """
        You are an expert taxonomist classifying technical skills.
        Your goal is to determine the parent L2 category/categories for a given L1 skill.
        L1 skills are specific tools, libraries, techniques, concepts, platforms, etc. (e.g., 'TensorFlow', 'Python', 'Regression Analysis', 'AWS S3').
        L2 categories are broader groupings like 'Frameworks', 'Tools', 'Platforms', 'Libraries', 'Concepts', 'Algorithms', etc. (e.g., 'ML Frameworks', 'Programming Languages', 'Statistical Modeling', 'Cloud Storage').
        The L1 skill must be a specific *instance* or *member* of the L2 category.
        Critically: Do NOT link an L1 skill to an L2 category if they are essentially synonyms or represent the same concept (e.g., do not link L1 'Vector Databases' to L2 'Vector Databases'; do not link L1 'Machine Learning' to L2 'Machine Learning Concepts').

        L1 Skill: "{l1_skill}"

        Candidate L2 Categories (Select ALL appropriate parent categories from this list ONLY):
        {candidate_list_str}

        Instructions:
        1. Carefully evaluate if the L1 skill is a specific example belonging to each candidate L2 category, following the synonym rule above.
        2. Select ALL L2 categories from the candidate list that are valid parent classifications.
        3. If NONE of the candidates are suitable (due to being synonyms or incorrect classifications), BUT you know another EXISTING L2 category in the broader taxonomy that IS suitable, provide its name as 'suggested_l2_category'. Only suggest if none of the candidates fit, and only suggest an *existing* category. Do not suggest synonyms.
        4. If no candidates fit and you cannot suggest a suitable existing L2 category, leave both 'selected_l2_categories' and 'suggested_l2_category' empty or null.

        Return ONLY a JSON object adhering to the provided schema with 'selected_l2_categories' (a list, possibly empty), 'suggested_l2_category' (string or null), and optionally 'reasoning'.
        Example 1 (Multiple Fits): {{"selected_l2_categories": ["ML Frameworks", "DL Frameworks"], "suggested_l2_category": null, "reasoning": "TensorFlow is both a general ML framework and specifically used for Deep Learning."}}
        Example 2 (One Fit): {{"selected_l2_categories": ["Programming Languages"], "suggested_l2_category": null, "reasoning": "Python is a programming language."}}
        Example 3 (Suggestion): {{"selected_l2_categories": [], "suggested_l2_category": "Cloud Storage", "reasoning": "S3 is a specific cloud storage service, none of the candidates fit well."}}
        Example 4 (No Fit/Synonym): {{"selected_l2_categories": [], "suggested_l2_category": null, "reasoning": "L1 'Vector Databases' is the category itself, not a specific instance."}}
        """
        candidate_list_str = "\n".join([f"- {c}" for c in candidate_l2s])
        inputs = {"l1_skill": l1_skill, "candidate_list_str": candidate_list_str}
        try:
            result_str = run_llm_chain(llm=self.judge_llm, template_str=judge_prompt, input_vars=["l1_skill", "candidate_list_str"], inputs=inputs, fail_softly=True)
            if result_str.startswith("[LLM FAILED]"): logger.error(f"Judge LLM call failed for '{l1_skill}': {result_str}"); return [], None
            json_str = extract_json_block(result_str)
            if not json_str: logger.error(f"No JSON from Judge LLM for '{l1_skill}'. Raw: {result_str}"); return [], None
            parsed_data = json.loads(json_str)
            validation_result = L1L2ValidationResult(**parsed_data)
            selected_categories = validation_result.selected_l2_categories
            suggested_category = validation_result.suggested_l2_category
            reasoning = validation_result.reasoning or "No reasoning."
            normalized_l1 = normalize_name(l1_skill)
            valid_selected = [cat for cat in selected_categories if cat in candidate_l2s and normalize_name(cat) != normalized_l1]
            if len(valid_selected) != len(selected_categories): logger.warning(f"Judge selected non-candidate or self-ref L2s for '{l1_skill}'. Original: {selected_categories}, Filtered: {valid_selected}")
            valid_suggestion = None
            if suggested_category and normalize_name(suggested_category) != normalized_l1:
                all_l2_names = self._get_all_l2_names()
                if suggested_category in all_l2_names: valid_suggestion = suggested_category; logger.info(f"Judge validation for '{l1_skill}': Suggested L2 = '{valid_suggestion}'. Reasoning: {reasoning}")
                else: logger.warning(f"Judge suggested non-existent L2 '{suggested_category}' for '{l1_skill}'. Ignoring.")
            elif suggested_category: logger.debug(f"Ignoring suggested self-reference L2 '{suggested_category}' for L1 '{l1_skill}'.")
            if valid_selected: logger.info(f"Judge validation for '{l1_skill}': Selected L2s = {valid_selected}. Reasoning: {reasoning}")
            return valid_selected, valid_suggestion
        except json.JSONDecodeError as e: logger.error(f"JSON parse error Judge LLM '{l1_skill}': {e}. JSON: '{json_str}'"); return [], None
        except ValidationError as e: logger.error(f"Schema validation error Judge LLM '{l1_skill}': {e}. Data: {parsed_data}"); return [], None
        except Exception as e: logger.error(f"Error during Judge LLM validation '{l1_skill}': {e}", exc_info=True); return [], None

    def _generate_and_create_l2s_for_new_l3(self, l3_name: str):
        logger.info(f"Proactively generating L2 categories for new L3 domain: '{l3_name}'")
        prompt = """
        You are an expert taxonomist. Given a high-level technical domain (L3), generate a list of standard, common L2 category names that typically fall under it. L2 categories represent groupings like 'Frameworks', 'Tools', 'Platforms', 'Libraries', 'Concepts', 'Algorithms', 'Protocols', 'Standards', etc. Keep the names concise and conventional. Aim for 3-7 standard categories relevant to the L3 domain.
        L3 Domain: "{l3_domain}"
        Return ONLY a JSON object adhering to the provided schema with the key 'l2_category_names' containing the list of generated L2 names. Example for L3="Cloud Computing": {{"l2_category_names": ["Cloud Platforms", "Cloud Storage", "Serverless Computing", "Infrastructure as Code", "Container Orchestration"]}} Example for L3="Cybersecurity": {{"l2_category_names": ["Security Frameworks", "Threat Detection Tools", "Compliance Standards", "Encryption Techniques", "Network Security"]}}
        """
        inputs = {"l3_domain": l3_name}
        try:
            result_str = run_llm_chain(llm=self.judge_llm, template_str=prompt, input_vars=["l3_domain"], inputs=inputs, fail_softly=True)
            if result_str.startswith("[LLM FAILED]"): logger.error(f"L2 generation LLM failed for L3 '{l3_name}': {result_str}"); return
            json_str = extract_json_block(result_str)
            if not json_str: logger.error(f"No JSON from L2 generation LLM for L3 '{l3_name}'. Raw: {result_str}"); return
            parsed_data = json.loads(json_str)
            result = GeneratedL2Categories(**parsed_data)
            new_l2_names = result.l2_category_names
            if not new_l2_names: logger.warning(f"LLM generated no L2 categories for new L3 '{l3_name}'."); return
            logger.info(f"LLM generated L2 categories for '{l3_name}': {new_l2_names}")
            category_list = [{"l2_name": l2, "l3_name": l3_name} for l2 in new_l2_names]
            query = "UNWIND $category_list as cat MERGE (l3:L3 {name: cat.l3_name}) MERGE (l2:L2 {name: cat.l2_name}) MERGE (l2)-[:PART_OF]->(l3) RETURN count(l2) as processed_count"
            self.connector.execute_query(query, parameters={"category_list": category_list})
            logger.info(f"Created {len(new_l2_names)} new L2 categories under L3 '{l3_name}'.")
            self._all_l2_names_cache = None; self._l2_repr_embedding_cache = None
        except Exception as e: logger.error(f"Error generating/creating L2s for new L3 '{l3_name}': {e}", exc_info=True)

    def _determine_l3_for_orphan(self, l1_skill: str) -> Optional[str]:
        if self._is_soft_skill(l1_skill):
             logger.info(f"Identified '{l1_skill}' as soft skill. Assigning to L3 'Soft Skills'.")
             self.initialize_l3_domains(["Soft Skills"])
             self.initialize_l2_categories({"Soft Skills": "Soft Skills"})
             return "Soft Skills"
        logger.debug(f"Determining L3 domain for orphan skill: '{l1_skill}'")
        all_l3_names = self._get_all_l3_names();
        if not all_l3_names: logger.error("No L3 domains found in graph."); return None
        prompt = """
        You are an expert taxonomist. Given the L1 skill and a list of existing high-level L3 domains, select the single most appropriate L3 domain for this skill. L1 Skill: "{l1_skill}" Existing L3 Domains:\n{l3_list_str}\nChoose the L3 domain that best represents the primary field or area the L1 skill belongs to. If none seem appropriate, suggest a NEW, concise, high-level L3 domain name suitable for this skill. Return ONLY a JSON object adhering to the provided schema with 'l3_domain' (either existing or new) and optionally 'reasoning'. Example (Existing): {{"l3_domain": "Cloud Computing", "reasoning": "AWS S3 is a cloud storage service."}} Example (New): {{"l3_domain": "Quantum Computing", "reasoning": "Qiskit is a quantum computing framework."}}
        """
        l3_list_str = "\n".join([f"- {name}" for name in sorted(list(all_l3_names))])
        inputs = {"l1_skill": l1_skill, "l3_list_str": l3_list_str}
        try:
            result_str = run_llm_chain(llm=self.judge_llm, template_str=prompt, input_vars=["l1_skill", "l3_list_str"], inputs=inputs, fail_softly=True)
            if result_str.startswith("[LLM FAILED]"): logger.error(f"L3 determination LLM failed for '{l1_skill}': {result_str}"); return None
            json_str = extract_json_block(result_str)
            if not json_str: logger.error(f"No JSON from L3 determination LLM for '{l1_skill}'. Raw: {result_str}"); return None
            parsed_data = json.loads(json_str)
            result = L3DeterminationResult(**parsed_data)
            determined_l3 = result.l3_domain
            if not determined_l3: logger.info(f"LLM could not determine suitable L3 for orphan '{l1_skill}'."); return None
            if determined_l3.lower() not in (name.lower() for name in all_l3_names):
                logger.info(f"Determined NEW L3 domain for orphan '{l1_skill}': {determined_l3}. Creating node and generating L2s...")
                create_l3_query = "MERGE (l3:L3 {name: $l3_name}) RETURN l3.name"
                create_result = self.connector.execute_query(create_l3_query, parameters={"l3_name": determined_l3})
                if not create_result: logger.error(f"Failed to create new L3 node '{determined_l3}'."); return None
                self._all_l3_names_cache = None
                self._generate_and_create_l2s_for_new_l3(determined_l3)
            else: logger.info(f"Determined existing L3 for orphan '{l1_skill}': {determined_l3}")
            return determined_l3
        except Exception as e: logger.error(f"Error determining L3 for orphan '{l1_skill}': {e}", exc_info=True); return None

    def _is_soft_skill(self, l1_skill: str) -> bool:
        soft_skill_keywords = ["communication", "leadership", "collaboration", "thinking", "learning", "ownership", "creativity", "mentorship", "innovation", "softskill"]
        return any(keyword in normalize_name(l1_skill) for keyword in soft_skill_keywords)

    def _determine_l2_name_for_orphan(self, l1_skill: str, l3_domain: str) -> Optional[str]:
        if l3_domain == "Soft Skills":
            logger.info(f"Assigning orphan soft skill '{l1_skill}' to L2 'Soft Skills'.")
            merge_l2_query = "MERGE (l3:L3 {name: 'Soft Skills'}) MERGE (l2:L2 {name: 'Soft Skills'}) MERGE (l2)-[:PART_OF]->(l3) RETURN l2.name"
            l2_result = self.connector.execute_query(merge_l2_query)
            if not l2_result: logger.error(f"Failed to merge fallback L2 category 'Soft Skills' for orphan skill '{l1_skill}'."); return None
            self._all_l2_names_cache = None; self._l2_repr_embedding_cache = None
            return "Soft Skills"
        logger.debug(f"Determining L2 name for orphan skill: '{l1_skill}' under L3: '{l3_domain}'")
        prompt = """
        You are an expert taxonomist. Given an L1 skill and its determined L3 domain, suggest a concise and appropriate L2 category name. L2 categories group related L1 skills (e.g., 'Frameworks', 'Libraries', 'Platforms', 'Tools', 'Concepts', 'Algorithms', 'Protocols', 'Standards'). L1 Skill: "{l1_skill}" L3 Domain: "{l3_domain}" Based on the L1 skill and its L3 domain, what is the most suitable L2 category name? Return ONLY a JSON object adhering to the provided schema with 'l2_category_name' and optionally 'reasoning'. Example: {{"l2_category_name": "DL Frameworks", "reasoning": "TensorFlow is a framework within Deep Learning."}}
        """
        inputs = {"l1_skill": l1_skill, "l3_domain": l3_domain}
        try:
            result_str = run_llm_chain(llm=self.judge_llm, template_str=prompt, input_vars=["l1_skill", "l3_domain"], inputs=inputs, fail_softly=True)
            if result_str.startswith("[LLM FAILED]"): logger.error(f"L2 name determination LLM failed for '{l1_skill}': {result_str}"); return None
            json_str = extract_json_block(result_str)
            if not json_str: logger.error(f"No JSON from L2 name determination LLM for '{l1_skill}'. Raw: {result_str}"); return None
            parsed_data = json.loads(json_str)
            result = L2NameDeterminationResult(**parsed_data)
            l2_name = result.l2_category_name
            if not l2_name: logger.info(f"LLM could not determine suitable L2 name for orphan '{l1_skill}'."); return None
            l2_name = l2_name.strip()
            if normalize_name(l2_name) == normalize_name(l1_skill): logger.warning(f"LLM suggested L2 '{l2_name}' which is a synonym for L1 '{l1_skill}'. Skipping."); return None
            logger.info(f"Determined L2 name for orphan '{l1_skill}': '{l2_name}' (under L3 '{l3_domain}')")
            merge_l2_query = "MERGE (l3:L3 {name: $l3_name}) MERGE (l2:L2 {name: $l2_name}) MERGE (l2)-[:PART_OF]->(l3) RETURN l2.name"
            l2_result = self.connector.execute_query(merge_l2_query, parameters={"l2_name": l2_name, "l3_name": l3_domain})
            if not l2_result: logger.error(f"Failed to merge fallback L2 category '{l2_name}' for orphan skill '{l1_skill}'."); return None
            self._all_l2_names_cache = None; self._l2_repr_embedding_cache = None
            return l2_name
        except Exception as e: logger.error(f"Error determining/merging L2 name for orphan '{l1_skill}': {e}", exc_info=True); return None

    def _link_l1_to_l2(self, l1_skill: str, l2_name: str):
        """Creates the BELONGS_TO relationship between an existing L1 and L2 node."""
        logger.debug(f"Creating link: L1 '{l1_skill}' -> L2 '{l2_name}'")
        link_query = "MATCH (l1:L1 {name: $l1_name}) MATCH (l2:L2 {name: $l2_name}) MERGE (l1)-[:BELONGS_TO]->(l2) RETURN l2.name as category_name"
        params = {"l1_name": l1_skill, "l2_name": l2_name}
        try:
            result = self.connector.execute_query(link_query, parameters=params)
            if result: logger.debug(f"Linked L1 '{l1_skill}' to L2 '{result[0]['category_name']}'")
            else: logger.warning(f"Failed MERGE relationship L1 '{l1_skill}' -> L2 '{l2_name}'.")
            return bool(result)
        except Exception as e: logger.error(f"Error merging relationship L1 '{l1_skill}' -> L2 '{l2_name}': {e}", exc_info=True); return False

    # --- Refactored Main Processing Methods ---
    def _normalize_skill_list(self, skills_to_process: List[str]) -> List[str]:
        """Normalizes a list of skills (lowercase, unique, stripped)."""
        if not skills_to_process: return []
        normalized_set = {s.lower().strip() for s in skills_to_process if s and s.strip()}
        return sorted(list(normalized_set))

    def _ensure_l1_nodes_exist(self, skills: List[str], embeddings: List[List[float]]):
        """Ensures L1 nodes exist for the given skills and updates embeddings."""
        if not skills or not embeddings or len(skills) != len(embeddings):
            logger.error("Mismatch between skills and embeddings list in _ensure_l1_nodes_exist.")
            return
        logger.info(f"Ensuring existence and updating embeddings for {len(skills)} L1 nodes...")
        for skill, embedding in zip(skills, embeddings):
            merge_l1_query = "MERGE (l1:L1 {name: $l1_name}) ON CREATE SET l1.embedding = $l1_embedding ON MATCH SET l1.embedding = $l1_embedding RETURN l1.name"
            try: self.connector.execute_query(merge_l1_query, parameters={"l1_name": skill, "l1_embedding": embedding})
            except Exception as e: logger.error(f"Error merging L1 node '{skill}': {e}")

    def _link_l1_to_validated_l2s(self, skill: str, embedding: List[float]) -> bool:
        """Finds candidates, validates, and links a single L1 skill. Returns True if linked."""
        linked = False
        candidate_matches = self._find_candidate_l2_matches(skill, embedding)
        candidate_l2_names = [name for name, score in candidate_matches]
        validated_l2_names, suggested_l2_name = self._validate_l1_l2_connections(skill, candidate_l2_names)
        target_l2_names = set(validated_l2_names)
        if suggested_l2_name: target_l2_names.add(suggested_l2_name)
        if target_l2_names:
            for l2_name in target_l2_names:
                if self._link_l1_to_l2(skill, l2_name): linked = True
        else: logger.debug(f"No validated or suggested L2 connections for '{skill}' in initial pass.")
        return linked

    def _process_skill_list_internal(self, skills_to_process: List[str], is_candidate: bool = False, is_jd: bool = False) -> Set[str]:
        """
        Internal method to process a list of *already standardized* skills.
        Ensures nodes exist, links them, and updates internal sets.
        """
        if not self.connector._driver or not self.embedder.model: return set()
        # Expects standardized_skills list as input now
        if not standardized_skills: return set()

        logger.info(f"Processing {len(standardized_skills)} standardized skills for graph update (Candidate: {is_candidate}, JD: {is_jd})...")

        # Update internal sets based on flags
        if is_candidate:
            self.candidate_skills_standardized.update(standardized_skills)
        if is_jd:
            # TODO: Decide if JD skills should also be standardized eventually
            # For now, assuming they might be passed pre-normalized or extracted directly
            self.jd_skills_standardized.update(standardized_skills)

        # --- Graph Operations with Standardized Skills ---
        l1_embeddings = self.embedder.get_embeddings(standardized_skills)
        if not l1_embeddings or len(standardized_skills) != len(l1_embeddings):
            logger.error("Failed embeddings for standardized skill list."); return set()

        self._ensure_l1_nodes_exist(standardized_skills, l1_embeddings)
        if not self._calculate_representative_l2_embeddings(): return set() # Ensure L2 embeddings are ready

        linked_l1_skills_in_batch = set()
        processed_count = 0
        for skill, embedding in zip(standardized_skills, l1_embeddings):
            processed_count += 1
            logger.debug(f"Attempting initial link for L1 skill ({processed_count}/{len(standardized_skills)}): '{skill}'")
            if self._link_l1_to_validated_l2s(skill, embedding):
                linked_l1_skills_in_batch.add(skill)

        logger.info(f"Finished initial graph linking for skill list. Linked {len(linked_l1_skills_in_batch)}/{len(standardized_skills)} standardized skills.")
        # Return the set of *standardized* skills that were successfully linked or ensured in the graph
        # Return the set of *standardized* skills that were successfully linked or ensured in the graph
        return linked_l1_skills_in_batch

    def process_candidate_skills(self, candidate_processor: CandidateSkillProcessor, force_process: bool = False):
        """
        Processes candidate skills obtained from the CandidateSkillProcessor.

        Args:
            candidate_processor: An initialized CandidateSkillProcessor instance.
            force_process: If True, force reprocessing of skills by the processor.
        """
        logger.info("Getting processed candidate skills...")
        # Get raw, standardized skills, and the map from the processor
        raw_skills, std_skills, std_map = candidate_processor.get_skills(force_process=force_process)

        # Store the map and the standardized skills in the builder
        self.skill_standardization_map = std_map
        self.candidate_skills_standardized = std_skills # Overwrite with the latest

        if not std_skills:
            logger.warning("No standardized candidate skills received from processor.")
            return

        # Process the standardized skills in the graph
        self._process_skill_list_internal(list(std_skills), is_candidate=True, is_jd=False)


    def process_job_description(self, job_description: str):
        """
        Extracts L1 skills from JD, processes them (without standardization for now),
        and updates the graph.
        """
        if not self.connector._driver or not self.embedder.model: return
        logger.info("Processing job description for skill graph update...")
        l1_skills = self.extractor.extract_skills(job_description)
        if not l1_skills: return
        self._process_skill_list_internal(l1_skills, is_jd=True)

    def update_skill_properties(self):
        """Sets source flags and match_score on L1 nodes."""
        # ... (implementation remains the same) ...
        if not self.connector._driver: return
        logger.info("Updating L1 node properties (source flags, match_score) using standardized skills...")
        reset_query = "MATCH (n:L1) REMOVE n.is_candidate_skill, n.is_jd_skill, n.match_score"
        self.connector.execute_query(reset_query)
        logger.debug("Reset L1 properties.")

        # Use standardized skill sets
        if self.candidate_skills_standardized:
            cand_query = "UNWIND $skills as skill_name MATCH (l1:L1 {name: skill_name}) SET l1.is_candidate_skill = true"
            self.connector.execute_query(cand_query, parameters={"skills": list(self.candidate_skills_standardized)})
            logger.info(f"Set is_candidate_skill=true for {len(self.candidate_skills_standardized)} standardized L1 nodes.")
        else:
            logger.info("No standardized candidate skills to mark.")

        if self.jd_skills_standardized:
            jd_query = "UNWIND $skills as skill_name MATCH (l1:L1 {name: skill_name}) SET l1.is_jd_skill = true"
            self.connector.execute_query(jd_query, parameters={"skills": list(self.jd_skills_standardized)})
            logger.info(f"Set is_jd_skill=true for {len(self.jd_skills_standardized)} standardized L1 nodes.")
        else:
             logger.info("No standardized JD skills to mark.")

        # Calculate score based on the flags set using standardized names
        score_query = """
        MATCH (l1:L1)
        SET l1.match_score = CASE
            WHEN l1.is_candidate_skill = true AND l1.is_jd_skill = true THEN 1
            ELSE 0
        END
        RETURN count(l1) as updated_count
        """
        result = self.connector.execute_query(score_query)
        if result: logger.info(f"Calculated match_score for {result[0]['updated_count']} L1 nodes.")
        else: logger.error("Failed to calculate match_score for L1 nodes.")

    def propagate_scores(self):
        """Calculates and sets l2_score and l3_score by propagating match_score."""
        # ... (implementation remains the same) ...
        if not self.connector._driver: return
        logger.info("Propagating scores up the hierarchy (L1 -> L2 -> L3)...")
        reset_l2l3 = "MATCH (n) WHERE n:L2 OR n:L3 REMOVE n.l2_score, n.l3_score"
        self.connector.execute_query(reset_l2l3)
        logger.debug("Reset L2/L3 scores.")
        l2_score_query = "MATCH (l1:L1)-[:BELONGS_TO]->(l2:L2) WHERE l1.match_score IS NOT NULL WITH l2, sum(l1.match_score) as total_l1_score SET l2.l2_score = total_l1_score RETURN count(l2) as updated_l2_count"
        l2_result = self.connector.execute_query(l2_score_query)
        if l2_result: logger.info(f"Calculated l2_score for {l2_result[0]['updated_l2_count']} L2 nodes.")
        else: logger.error("Failed to calculate l2_score.")
        l3_score_query = "MATCH (l2:L2)-[:PART_OF]->(l3:L3) WHERE l2.l2_score IS NOT NULL WITH l3, sum(l2.l2_score) as total_l2_score SET l3.l3_score = total_l2_score RETURN count(l3) as updated_l3_count"
        l3_result = self.connector.execute_query(l3_score_query)
        if l3_result: logger.info(f"Calculated l3_score for {l3_result[0]['updated_l3_count']} L3 nodes.")
        else: logger.error("Failed to calculate l3_score.")

    def consolidate_all_links(self):
        """Checks all L1 nodes against all L2 categories and adds missing valid links."""
        # ... (implementation remains the same) ...
        if not self.connector._driver or not self.embedder.model: return
        logger.info("Consolidating all L1 -> L2 links...")
        l1_query = "MATCH (l1:L1) WHERE l1.embedding IS NOT NULL RETURN l1.name as name, l1.embedding as embedding"
        all_l1_data = self.connector.execute_query(l1_query)
        if not all_l1_data: logger.warning("No L1 nodes with embeddings found for consolidation."); return
        rel_query = "MATCH (l1:L1)-[r:BELONGS_TO]->(l2:L2) RETURN l1.name as l1_name, l2.name as l2_name"
        existing_rels_raw = self.connector.execute_query(rel_query)
        existing_rels = set((rel['l1_name'], rel['l2_name']) for rel in existing_rels_raw)
        logger.info(f"Found {len(existing_rels)} existing L1->L2 relationships.")
        if not self._calculate_representative_l2_embeddings(): return
        consolidation_links_created = 0
        for l1_node in all_l1_data:
            skill = l1_node['name']
            embedding = l1_node['embedding']
            if not embedding: continue
            candidate_matches = self._find_candidate_l2_matches(skill, embedding)
            candidate_l2_names = [name for name, score in candidate_matches]
            new_candidates = [l2 for l2 in candidate_l2_names if (skill, l2) not in existing_rels]
            if not new_candidates: continue
            validated_l2_names, suggested_l2_name = self._validate_l1_l2_connections(skill, new_candidates)
            target_l2_names = set(validated_l2_names)
            if suggested_l2_name and (skill, suggested_l2_name) not in existing_rels: target_l2_names.add(suggested_l2_name)
            if target_l2_names:
                for l2_name in target_l2_names:
                    if self._link_l1_to_l2(skill, l2_name): consolidation_links_created += 1; existing_rels.add((skill, l2_name))
        logger.info(f"Consolidation finished. Created {consolidation_links_created} new L1->L2 relationships.")

    def link_orphans(self):
        """Finds L1 nodes with no L2 parents and attempts to link them using fallback."""
        # ... (implementation remains the same) ...
        if not self.connector._driver or not self.embedder.model: return
        logger.info("Checking for and linking orphan L1 skills...")
        orphan_query = "MATCH (l1:L1) WHERE NOT (l1)-[:BELONGS_TO]->(:L2) RETURN l1.name as name"
        orphans = self.connector.execute_query(orphan_query)
        if not orphans: logger.info("No orphan L1 skills found."); return
        logger.info(f"Found {len(orphans)} orphan L1 skills. Attempting fallback linking...")
        fallback_linked_count = 0
        for orphan in orphans:
            skill = orphan['name']
            logger.debug(f"Processing orphan skill: '{skill}'")
            l3_domain = self._determine_l3_for_orphan(skill)
            if not l3_domain: continue
            l2_name = self._determine_l2_name_for_orphan(skill, l3_domain)
            if not l2_name: continue
            if self._link_l1_to_l2(skill, l2_name): fallback_linked_count += 1
        logger.info(f"Fallback linking finished. Linked {fallback_linked_count}/{len(orphans)} orphan skills.")

    def close_connection(self):
        """Closes the Neo4j connection."""
        self.connector.close()

# --- Test Execution ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("--- Running SkillGraphBuilder Test ---")
    try:
        builder = SkillGraphBuilder()
        # Clear graph for clean test run (optional)
        # logger.warning("Clearing existing graph data...")
        # builder.connector.execute_query("MATCH (n) DETACH DELETE n")
        # builder._all_l2_names_cache = None; builder._all_l3_names_cache = None; builder._l2_repr_embedding_cache = None
        # builder.candidate_skills.clear(); builder.jd_skills.clear()

        builder.initialize_l3_domains()
        builder.initialize_l2_categories()
        sample_jd = """
        We are seeking a Senior Machine Learning Engineer to join our dynamic team.
        Responsibilities include designing, developing, and deploying machine learning models using Python and TensorFlow.
        Experience with AWS services (S3, SageMaker) and Docker is required.
        Familiarity with NLP techniques and libraries like spaCy or NLTK is a plus.
        Must have strong experience with SQL databases and data warehousing concepts.
        Knowledge of CI/CD pipelines using Jenkins or GitLab CI is essential.
        Experience with Langchain and vector databases like Pinecone is highly desirable.
        """
        # Simulate processing skills from skills.txt first
        skills_from_file = ["PyTorch", "Kubernetes", "React", "GCP", "Kafka", "Airflow", "Scrum", "Vector Databases", "Python", "SQL", "Leadership", "Communication Skills"]
        builder.process_skill_list(skills_from_file)

        # Process JD
        builder.process_job_description(sample_jd)

        # Update properties and scores
        builder.update_skill_properties()

        # Consolidate links
        builder.consolidate_all_links()

        # Link orphans
        builder.link_orphans()

        # Propagate scores
        builder.propagate_scores()

        # Test: List ALL L1 nodes created with flags/scores
        logger.info("--- Test: Listing ALL L1 Skills with properties ---")
        list_all_l1_query = "MATCH (l1:L1) RETURN l1.name as name, l1.is_candidate_skill as cand, l1.is_jd_skill as jd, l1.match_score as score ORDER BY l1.name"
        all_l1_nodes = builder.connector.execute_query(list_all_l1_query)
        if all_l1_nodes:
            print(f"\nFound {len(all_l1_nodes)} L1 Skills in total:")
            for node in all_l1_nodes: print(f"- {node['name']} (Cand: {node.get('cand', False)}, JD: {node.get('jd', False)}, Score: {node.get('score', 0)})")
        else: print("\nCould not retrieve L1 skills (or none exist). Check logs.")

        # Test: List L2/L3 nodes with scores
        logger.info("--- Test: Listing L2/L3 Scores ---")
        list_l2_scores = "MATCH (l2:L2) WHERE l2.l2_score > 0 RETURN l2.name as name, l2.l2_score as score ORDER BY l2.l2_score DESC"
        l2_nodes = builder.connector.execute_query(list_l2_scores)
        if l2_nodes:
            print("\nL2 Nodes with Scores > 0:")
            for node in l2_nodes: print(f"- {node['name']}: {node['score']}")
        else: print("\nNo L2 nodes with scores > 0 found.")

        list_l3_scores = "MATCH (l3:L3) WHERE l3.l3_score > 0 RETURN l3.name as name, l3.l3_score as score ORDER BY l3.l3_score DESC"
        l3_nodes = builder.connector.execute_query(list_l3_scores)
        if l3_nodes:
            print("\nL3 Nodes with Scores > 0:")
            for node in l3_nodes: print(f"- {node['name']}: {node['score']}")
        else: print("\nNo L3 nodes with scores > 0 found.")

        # Test: Verify all L1 nodes have at least one parent
        logger.info("--- Test: Verifying all L1 nodes have parents ---")
        orphan_check_query = "MATCH (l1:L1) WHERE NOT (l1)-[:BELONGS_TO]->(:L2) RETURN l1.name as orphan_name"
        orphans = builder.connector.execute_query(orphan_check_query)
        if not orphans: print("\nVerification PASSED: All L1 nodes have at least one L2 parent.")
        else: print(f"\nVerification FAILED: Found {len(orphans)} orphan L1 nodes: {[o['orphan_name'] for o in orphans]}")

        builder.close_connection()
        logger.info("--- SkillGraphBuilder Test Finished ---")
    except ConnectionError as e: logger.error(f"Test failed due to Neo4j connection error: {e}")
    except Exception as e: logger.error(f"An unexpected error occurred during the test: {e}", exc_info=True)
