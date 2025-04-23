from resume_tailor.tailors.skills import jd_scoring as jds 
from resume_tailor.tailors.skills import emedding_skills as es
from resume_tailor.tailors.skills import db_skill_matcher as dsm
from resume_tailor.tailors.skills import utils as utl
from resume_tailor.tailors.skills import skill_classification as sc
import json
import ast


class SkillsRefiner:
    """Refines the skills section of a resume using LLM-guided multi-step processing."""

    MAX_REFINEMENT_ITERATIONS = 5
    log_file = "skills_refinement.log"

    def __init__(self, raw_client, model_name, job_description):
        self.raw_client = raw_client
        self.model_name = model_name
        self.jd = job_description
    
    def skill_matching(self):
        skills_dict = jds.scored_skill_extractor(self.raw_client, self.model_name, self.jd)
        
        # creating vector DB
        vector = es.vector_db()
        
        skill_classification = dsm.classify_and_merge_skills(
            skills_dict=skills_dict,
            vectorstore=vector,
            threshold_exact=0.98,     # Exact match threshold
            threshold_strong=0.9,       # Strong match threshold
            threshold_weak=0.8,         # Weak match threshold
            initial_k=10,
            max_k=200
        )
        
        combined_missing, combined_candidates = self.combining_missing_skills(skill_classification)
        resolved_mapping = self.matching_missed_skills(combined_missing, combined_candidates)
        print(resolved_mapping)
        skill_classification = self.updating_skills(skill_classification, resolved_mapping)
        return skill_classification
        
        
            
    def combining_missing_skills(self, skill_classification):
        # Combine missing skills for "Primary" and "Inferred" JD classes.
        missing_primary = skill_classification.get("missing_skills_report", {}).get("Primary", [])
        missing_inferred = skill_classification.get("missing_skills_report", {}).get("Inferred", [])
        combined_missing = list(set(missing_primary + missing_inferred))

        # Build a prompt for the LLM.
        # Here we provide context by including the candidate skills from the "Primary" and "Inferred" buckets.
        candidate_primary = skill_classification.get("candidate_primary", [])
        candidate_inferred = skill_classification.get("candidate_inferred", [])
        combined_candidates = list(set(candidate_primary).union(set(candidate_inferred)))
        
        
        return combined_missing, combined_candidates

    def matching_missed_skills(self, combined_missing, combined_candidates):
        
        # Sanitize missing skills.
        sanitized_missing, missing_mapping = utl.sanitize_skills_with_mapping(combined_missing)
        
        tool_schema = sc.generate_missing_skills_tool_schema(sanitized_missing)
        
        prompt = (
            "You are a candidate skill resolver.\n\n"
            "The candidate is known to possess the following technical skills:\n"
            f"{json.dumps(combined_candidates)}\n\n"
            "Now, for each of the following **missing skills**, your task is to estimate how likely the candidate is to already know them, "
            "based on logical inference from their known skills.\n\n"
            "A skill should be rated as HIGH (close to 100) if it is:\n"
            "- commonly used alongside the known skills\n"
            "- required to work with or implement the known skills\n"
            "- part of the same framework, library, or domain cluster\n\n"
            "Rate each skill from 0 to 100:\n"
            "- 100 = extremely likely (directly implied by known skills)\n"
            "- 50 = moderately likely (common co-occurrence or weak implication)\n"
            "- 0 = very unlikely (no clear relation)\n\n"
            "**Be especially mindful of multi-skill combinations that imply deeper knowledge.**\n"
            "Use the structured tool provided to return your answer as JSON. Only include the required keys from the schema. "
            "Do not include explanations — only the JSON output."
        )

        response = self.raw_client.messages.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            tools=tool_schema,  # Passing our tool schema here.
            max_tokens=1024,
            temperature=0.1,
            tool_choice={"type": "tool", "name": "missing_skills_checker"}
        )
        
        raw_output = response.content[0].input
        
        # If the raw output is a string, convert it to a dictionary.
        if isinstance(raw_output, str):
            try:
                raw_output = ast.literal_eval(raw_output)
            except Exception as e:
                raise ValueError(f"Error converting raw_output to dict: {e}")
        
        # Now process each key-value pair.
        resolved_mapping = {}
        for sanitized_key, value in raw_output.items():
            # Ensure the value is an int.
            if isinstance(value, int):
                final_value = value
            elif isinstance(value, float) and value.is_integer():
                final_value = int(value)
            else:
                try:
                    final_value = int(value)
                except Exception as e:
                    raise ValueError(f"Value for key '{sanitized_key}' is not an integer: {value}, error is: {e}")
            
            # Map the sanitized key back to the original skill name.
            original_skill = missing_mapping.get(sanitized_key, sanitized_key)
            resolved_mapping[original_skill] = final_value
        
        return resolved_mapping

    
    def updating_skills(self, skill_classification, resolved_mapping):
        for i in range(len(skill_classification["missing_skills_report"]["Primary"]) - 1, -1 , -1):
            skill = skill_classification["missing_skills_report"]["Primary"][i]
            if resolved_mapping.get(skill, 0) >= 80:
                skill_classification["candidate_primary"].append(skill)
                skill_classification["matched_but_not_exact_report"]["Primary"].append(skill)
                del skill_classification["missing_skills_report"]["Primary"][i]

        for i in range(len(skill_classification["missing_skills_report"]["Inferred"]) - 1, -1 , -1):
            skill = skill_classification["missing_skills_report"]["Inferred"][i]
            if resolved_mapping.get(skill, 0) >= 80:
                skill_classification["candidate_inferred"].append(skill)
                skill_classification["matched_but_not_exact_report"]["Inferred"].append(skill)
                del skill_classification["missing_skills_report"]["Inferred"][i]
        
        return skill_classification

                
