from resume_tailor.tailors.skills import skills_extractor as se
from resume_tailor.tailors.skills import header_extractor as he
from resume_tailor.tailors.skills import skill_header_mapper as shm
from resume_tailor.tailors.skills import utils as utl
from resume_tailor.tailors.skills.llm_response_validator import validate_and_fix_skill_mapping
from importlib import reload

class SkillHeaderGrouping:
    """Refines the skills section of a resume using LLM-guided multi-step processing."""

    MAX_REFINEMENT_ITERATIONS = 5
    log_file = "skills_refinement.log"

    def __init__(self, graph, raw_client, model_name, job_description):
        self.raw_client = raw_client
        self.model_name = model_name
        self.jd = job_description
        self.graph = graph

    def information_extraction(self,):
        skill_refiner = se.SkillsRefiner(self.raw_client, self.model_name, self.job_description)
        skill_classification = skill_refiner.skill_matching()

        primary_skills = [(i, 3) for i in skill_classification["candidate_primary"]]
        inferred_skills = [(i, 2) for i in skill_classification["candidate_inferred"]]
        skill_set_ranks = primary_skills + inferred_skills

        skill_set = [i[0] for i in skill_set_ranks]
        headers = he.extract_all_headers(self.raw_client, self.model_name, self.jd)
        
        new_headers, new_skills, old_skills, all_headers = self.get_layer_elements(headers, skill_set)
        all_skills_mapping = self.skill_header_mapper(new_headers, new_skills, old_skills, all_headers)
        
        reviewed_skill_mapping = self.feedback_loop_revision(all_skills_mapping)
        
        
        return reviewed_skill_mapping
        
    def get_layer_elements(self, headers, skills):
        headers_nodes = self.graph.get_level_nodes(label="L2")
        skill_nodes = self.graph.get_level_nodes(label="L1")

        headers_in = [i["name"].replace("_", " ") for i in headers_nodes]
        skills_in = [i["name"] for i in skill_nodes]

        new_headers = list(set(headers) - set(headers_in))
        all_headers = list(set(set(headers) | set(headers_in)))
        new_skills = list(set(skills) - set(skills_in))
        
        return new_headers, new_skills, skills_in, all_headers
    
    def new_skill_header_finder(self, new_headers, new_skills, old_skills, all_headers):
        
        final_new_skills_mapping = self.skill_header_mapping(all_headers, new_skills)
        final_old_skills_mapping = self.skill_header_mapping(new_headers, old_skills)
        
        for header in final_new_skills_mapping.keys():
            final_new_skills_mapping[header] = list(set(final_new_skills_mapping[header]) | set(final_old_skills_mapping.get(header, [])))
        
        return final_new_skills_mapping
        #TODO
        # check if there is any new skills that has no class
        all_skills = set()
        for _, arr in final_new_skills_mapping.items():
            all_skills |= set(arr)

        all_skills = [i.lower() for i in all_skills]
        check_skills = [i.lower() for i in new_skills]
    
    def skill_header_mapping(self, headers, skills):
        reload(shm)
        sanitized_headers, mapping = utl.sanitize_skills_with_mapping(headers)
        skills_mapping = shm.new_skills_to_all_headers(self.raw_client, self.model_name, skills, sanitized_headers)
        
        final_skills_mapping = {}
        for sanitized_key, arr in skills_mapping.items():
            # Map the sanitized key back to the original skill name.
            original_skill = mapping.get(sanitized_key, sanitized_key)
            final_skills_mapping[original_skill] = arr
        
        return final_skills_mapping
    
    def feedback_loop_revision(self, skill_mapping, max_iter=5):
        score = 50
        i = 0
        reload(shm)
        while score < 90 and i < max_iter:
            print(f"\n🔁 Iteration {i+1}")

            response = shm.judge_mapping_feedback_with_score(self.raw_client, self.model_name, skill_mapping)
            score = response["grade"]
            feedback = response["feedback"]

            print(f"🧠 Feedback Score: {score}")
            print(f"📌 Feedback: {feedback}")

            previous_mapping = skill_mapping.copy()

            # Retry loop for reclassification
            attempts = 0
            iteration_response = {}
            while not iteration_response and attempts < 3:
                try:
                    raw_response = shm.reclassify(self.raw_client, self.model_name, skill_mapping, feedback)
                    print(raw_response)
                    iteration_response = validate_and_fix_skill_mapping(raw_response)
                except Exception as e:
                    print(f"⚠️ Reclassification failed: {e}")
                    attempts += 1

            if not iteration_response:
                print("❌ Failed to reclassify after 3 attempts.")
                break

            skill_mapping = iteration_response
            i += 1

        print(f"\n🎉 Final mapping score: {score}")
        return skill_mapping