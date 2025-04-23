import numpy as np
from typing import List, Dict, Set, Any
from langchain.vectorstores.base import VectorStoreRetriever

# ------------------------
# Helper: Cosine Similarity
# ------------------------
def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    v1 = np.array(v1)
    v2 = np.array(v2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

# ------------------------
# Helper: Dynamic FAISS Search for a Single JD Skill
# ------------------------
def search_matches_for_skill(
    jd_skill: str,
    vectorstore: VectorStoreRetriever,
    threshold_exact: float,
    threshold_strong: float,
    threshold_weak: float,
    initial_k: int,
    max_k: int
) -> Dict[str, Set[str]]:
    """
    For a given JD skill, performs dynamic top-k search using the vectorstore.
    Classifies candidate results into three sets:
      - "complete_match": candidate skills with score ≥ threshold_exact.
      - "strong_match": candidate skills with threshold_strong ≤ score < threshold_exact.
      - "weak_match": candidate skills with threshold_weak ≤ score < threshold_strong.
    Stops early if the top result's score is below threshold_weak.
    """
    k_current = initial_k
    complete_match: Set[str] = set()
    strong_match: Set[str] = set()
    weak_match: Set[str] = set()
    
    while k_current <= max_k:
        results = vectorstore.similarity_search_with_score(jd_skill, k=k_current)
        if not results:
            break
        
        # If the top result is below weak threshold, no further search is needed.
        if results[0][1] < threshold_weak:
            break
        
        for doc, score in results:
            if score < threshold_weak:
                break  # Remaining scores are lower.
            if score >= threshold_exact:
                complete_match.add(doc.page_content)
            elif score >= threshold_strong:
                strong_match.add(doc.page_content)
            elif score >= threshold_weak:
                weak_match.add(doc.page_content)
        
        # If any relevant match was found, stop expanding.
        if complete_match or strong_match or weak_match:
            break
        else:
            k_current *= 2  # Expand the search scope.
    
    return {
        "complete_match": complete_match,
        "strong_match": strong_match,
        "weak_match": weak_match,
    }

# ------------------------
# Helper: Assign Matches to Candidate Buckets & Log JD Skills
# ------------------------
def assign_matches_to_buckets(
    jd_class: str,
    jd_skill: str,
    matches: Dict[str, Set[str]],
    buckets: Dict[str, Set[str]],
    report: Dict[str, Set[str]]
) -> None:
    """
    Assigns candidate results to buckets based on the JD class and logs the JD skill if there’s a strong match.
    
    For "Primary":
      - Exact matches go to candidate_primary.
      - If strong match exists:
            * Add the JD skill itself into candidate_primary,
            * Log the JD skill in the report for Primary,
            * And add the candidate strong-match results to candidate_inferred.
      - Weak matches go to candidate_inferred_sec.
    
    For "Inferred":
      - Exact matches go to candidate_inferred.
      - If strong match exists:
            * Add the JD skill to candidate_inferred,
            * Log it in the report for Inferred,
            * And add the candidate strong-match results to candidate_inferred_sec.
      - Weak matches go to candidate_inferred_sec.
    
    For "Inferred_Secondary":
      - All matches (exact, strong, weak) go to candidate_inferred_sec.
      - If a strong match exists, log the JD skill in the report for Inferred_Secondary.
    """
    if jd_class == "Primary":
        buckets["candidate_primary"].update(matches["complete_match"])
        if matches["strong_match"]:
            buckets["candidate_primary"].add(jd_skill)
            report["Primary"].add(jd_skill)
            buckets["candidate_inferred"].update(matches["strong_match"])
        buckets["candidate_inferred_sec"].update(matches["weak_match"])
        
    elif jd_class == "Inferred":
        buckets["candidate_inferred"].update(matches["complete_match"])
        if matches["strong_match"]:
            buckets["candidate_inferred"].add(jd_skill)
            report["Inferred"].add(jd_skill)
            buckets["candidate_inferred_sec"].update(matches["strong_match"])
        buckets["candidate_inferred_sec"].update(matches["weak_match"])
    
    elif jd_class == "Inferred_Secondary":
        buckets["candidate_inferred_sec"].update(matches["complete_match"] | matches["strong_match"] | matches["weak_match"])
        if matches["strong_match"]:
            report["Inferred_Secondary"].add(jd_skill)

# ------------------------
# Helper: Deduplicate Candidate Buckets by Priority
# ------------------------
def deduplicate_buckets(buckets: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """
    Deduplicates candidate buckets in the following priority:
      candidate_primary > candidate_inferred > candidate_inferred_sec.
    """
    final_candidate_primary = buckets["candidate_primary"]
    final_candidate_inferred = buckets["candidate_inferred"] - final_candidate_primary
    final_candidate_inferred_sec = buckets["candidate_inferred_sec"] - final_candidate_primary - final_candidate_inferred
    return {
        "candidate_primary": final_candidate_primary,
        "candidate_inferred": final_candidate_inferred,
        "candidate_inferred_sec": final_candidate_inferred_sec,
    }

# ------------------------
# Main Function: classify_and_merge_skills
# ------------------------
def classify_and_merge_skills(
    skills_dict: Dict[str, List[str]],         # JD skills organized by class.
                                       # Expected keys: "Primary", "Inferred", "Inferred_Secondary"
    vectorstore: VectorStoreRetriever, # FAISS vector store (must support similarity_search_with_score)
    threshold_exact: float = 0.98,
    threshold_strong: float = 0.9,
    threshold_weak: float = 0.8,
    initial_k: int = 10,
    max_k: int = 200
) -> Dict[str, Any]:
    """
    Processes JD skills per class, fetches candidate matches, assigns them to candidate buckets according
    to the following rules:
    
      For JD class "Primary":
          - Exact matches (>= threshold_exact) go to candidate_primary.
          - If strong matches (>= threshold_strong but < threshold_exact) exist:
                * Add the original JD skill to candidate_primary (upgrade it) and log it in the report.
                * Add the candidate strong-match results to candidate_inferred.
          - Weak matches (>= threshold_weak but < threshold_strong) go to candidate_inferred_sec.
    
      For JD class "Inferred":
          - Exact matches go to candidate_inferred.
          - If strong matches exist:
                * Add the JD skill to candidate_inferred and log it in the report.
                * Add the candidate strong-match results to candidate_inferred_sec.
          - Weak matches go to candidate_inferred_sec.
    
      For JD class "Inferred_Secondary":
          - All matches go to candidate_inferred_sec.
          - If strong matches exist, log the JD skill in the report.
    
    Finally, for each JD class, if the original JD skill is not found in its corresponding candidate bucket,
    it is recorded as missing in the missing_skills_report for that JD class.
    
    Deduplicates candidate buckets with priority: Primary > Inferred > Inferred_sec.
    
    Returns a dictionary containing:
      - candidate_primary: Sorted list.
      - candidate_inferred: Sorted list.
      - candidate_inferred_sec: Sorted list.
      - matched_but_not_exact_report: Per JD class.
      - missing_skills_report: Per JD class.
    """
    allowed_jd_classes = {"Primary", "Inferred", "Inferred_Secondary"}
    
    # Initialize candidate buckets.
    buckets = {
        "candidate_primary": set(),
        "candidate_inferred": set(),
        "candidate_inferred_sec": set(),
    }
    
    # Initialize strong match reports.
    matched_but_not_exact_report = {
        "Primary": set(),
        "Inferred": set(),
        "Inferred_Secondary": set(),
    }
    
    # Initialize missing skills report.
    missing_skills_report = {
        "Primary": set(),
        "Inferred": set(),
        "Inferred_Secondary": set(),
    }
    
    for jd_class, skill_arr in skills_dict.items():
        if jd_class not in allowed_jd_classes:
            continue
        
        for jd_skill in skill_arr:
            # Retrieve candidate matches for this JD skill.
            matches = search_matches_for_skill(
                jd_skill,
                vectorstore,
                threshold_exact,
                threshold_strong,
                threshold_weak,
                initial_k,
                max_k
            )
            
            # If no candidate match (exact, strong, or weak) is found, record the JD skill as missing.
            if not (matches["complete_match"] or matches["strong_match"] or matches["weak_match"]):
                missing_skills_report[jd_class].add(jd_skill)
                continue
            
            # Assign matches to candidate buckets based on JD class.
            assign_matches_to_buckets(jd_class, jd_skill, matches, buckets, matched_but_not_exact_report)
            
            # Updated missing check: only check the candidate bucket corresponding directly to the JD class.
            if jd_class == "Primary":
                if jd_skill not in buckets["candidate_primary"]:
                    missing_skills_report[jd_class].add(jd_skill)
            elif jd_class == "Inferred":
                if jd_skill not in buckets["candidate_inferred"]:
                    missing_skills_report[jd_class].add(jd_skill)
            elif jd_class == "Inferred_Secondary":
                if jd_skill not in buckets["candidate_inferred_sec"]:
                    missing_skills_report[jd_class].add(jd_skill)
    
    # Final deduplication based on priority.
    final_buckets = deduplicate_buckets(buckets)
    
    return {
        "candidate_primary": sorted(final_buckets["candidate_primary"]),
        "candidate_inferred": sorted(final_buckets["candidate_inferred"]),
        "candidate_inferred_sec": sorted(final_buckets["candidate_inferred_sec"]),
        "matched_but_not_exact_report": {
            "Primary": sorted(matched_but_not_exact_report["Primary"]),
            "Inferred": sorted(matched_but_not_exact_report["Inferred"]),
            "Inferred_Secondary": sorted(matched_but_not_exact_report["Inferred_Secondary"]),
        },
        "missing_skills_report": {
            "Primary": sorted(missing_skills_report["Primary"]),
            "Inferred": sorted(missing_skills_report["Inferred"]),
            "Inferred_Secondary": sorted(missing_skills_report["Inferred_Secondary"]),
        },
    }