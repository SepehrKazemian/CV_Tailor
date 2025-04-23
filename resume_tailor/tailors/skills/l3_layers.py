from resume_tailor.tailors.skills import skills_extractor as se
from resume_tailor.tailors.skills import header_extractor as he
from resume_tailor.tailors.skills import skill_header_mapper as shm
from resume_tailor.tailors.skills import utils as utl
from resume_tailor.tailors.skills import tag_canonizer as tc
from resume_tailor.tailors.skills import utils as utl

class GroupAliasNodeCreation:
    """Refines the skills section of a resume using LLM-guided multi-step processing."""

    MAX_REFINEMENT_ITERATIONS = 5
    log_file = "skills_refinement.log"

    def __init__(self, graph, raw_client, model_name, job_description):
        self.raw_client = raw_client
        self.model_name = model_name
        self.jd = job_description
        self.graph = graph
    
    def group_alias(self):
        shg = tc.SkillHeaderGrouping(self.graph, self.raw_client, self.model_name, self.jd)
        skill_group_mapping = shg.information_extraction()
        initial_group_alias = tc.tag_matching_main(skill_group_mapping)
    
    
    def llm_alias_validation(self, initial_group_alias):
        sanitized_headers, mapping = utl.sanitize_headers_with_mapping(initial_group_alias)
