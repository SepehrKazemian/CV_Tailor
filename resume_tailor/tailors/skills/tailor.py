import logging
import json
from typing import Dict, List, Optional, Tuple

from resume_tailor.utils.llm_provider import get_llm
from resume_tailor.utils.llm_utils import run_llm_chain
from resume_tailor.utils.prompt_templates import (
    STEP1_EXTRACT_SKILLS_PROMPT,
    STEP2_SELECT_HEADERS_PROMPT,
    STEP4_REFINE_HEADERS_PROMPT
)
from resume_tailor.tailors.tailor_utils import SKILL_SECTION_HEADERS
from resume_tailor.tailors.skills.llm_judge import JudgeEvaluation, get_judge_tool_schema
from resume_tailor.tailors.skills.logger import get_logger
from utils.json_extractor import extract_json_block
from resume_tailor.tailors.skills.skill_matcher import SkillMatcher
from resume_tailor.tailors.skills.skill_header_selector import SkillHeaderSelector

class SkillsRefiner:
    """Refines the skills section of a resume using LLM-guided multi-step processing."""

    MAX_REFINEMENT_ITERATIONS = 5
    log_file = "skills_refinement.log"

    def __init__(self, llm_provider: str, api_key: Optional[str] = None):
        self.model_name, self.llm, self.raw_llm_client = get_llm(
            llm_provider, api_key, return_raw=True
        )
        self.supports_tool_calling = (
            self.raw_llm_client and hasattr(self.raw_llm_client, 'messages') and
            hasattr(self.raw_llm_client.messages, 'create')
        )
        self.logger = get_logger()
        self.logger.info(f"Initialized SkillsRefiner with model: {self.model_name}")

        if not self.supports_tool_calling:
            self.logger.warning("Raw LLM client does not support tool calling. Judge step might fail.")

    def skills_pipeline(self, job_description, original_skills_text):
        
        Step1_llm = SkillMatcher(self.raw_llm_client, self.logger, self.model_name)
        Step2_llm = SkillHeaderSelector(self.llm, self.logger)
        
        header_selector_result = Step1_llm.extract_skills_grouped_by_header(job_description, original_skills_text)
        matched_skills = {key:header_selector_result[key]["matched_skills"] for key in header_selector_result.keys()}
        
        top_headers_skills = Step2_llm.select_top_headers(matched_skills)
        
        
        
        
        
    def _step1_extract_skills_per_header(
        self, job_description: str, original_skills_text: str
    ) -> Dict[str, Dict[str, List[str]]]:
        """ Step 1: Extracts skills per header using LLM calls. """
        self.logger.info("Starting Step 1: Extracting skills per header.")
        all_skills_data = {}

        for header in SKILL_SECTION_HEADERS:
            self.logger.debug(f"Processing header: {header}")
            inputs = {
                "header": header,
                "job_description": job_description,
                "original_skills_text": original_skills_text
            }
            input_vars = list(inputs.keys())
            try:
                result_str = run_llm_chain(
                    self.llm, STEP1_EXTRACT_SKILLS_PROMPT, input_vars, inputs, fail_softly=True
                )
                self.logger.debug(f"LLM Raw Output for '{header}':\n{result_str}")

                if result_str.startswith("[LLM FAILED]"):
                    self.logger.warning(f"LLM failed for header '{header}': {result_str}")
                    all_skills_data[header] = {"job_skills": [], "candidate_skills": []}
                    continue

                json_str = extract_json_block(result_str)
                if not json_str:
                    self.logger.warning(f"Could not extract JSON block for header '{header}'. Raw: {result_str}")
                    all_skills_data[header] = {"job_skills": [], "candidate_skills": []}
                    continue

                try:
                    skills_dict = json.loads(json_str)
                    # Basic validation
                    if isinstance(skills_dict, dict) and \
                       "job_skills" in skills_dict and isinstance(skills_dict["job_skills"], list) and \
                       "candidate_skills" in skills_dict and isinstance(skills_dict["candidate_skills"], list):
                        all_skills_data[header] = skills_dict
                        self.logger.debug(f"Parsed skills for '{header}': {skills_dict}")
                    else:
                        self.logger.warning(f"Invalid JSON structure received for header '{header}': {json_str}")
                        all_skills_data[header] = {"job_skills": [], "candidate_skills": []}
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to parse extracted JSON for header '{header}': {json_str}")
                    all_skills_data[header] = {"job_skills": [], "candidate_skills": []}

            except Exception as e:
                self.logger.error(f"Error processing header '{header}': {e}", exc_info=True)
                all_skills_data[header] = {"job_skills": [], "candidate_skills": []}

        self.logger.info("Finished Step 1: Skill extraction per header.")
        self.logger.debug(f"Step 1 Full Output: {json.dumps(all_skills_data, indent=2)}")
        return all_skills_data

    def _step2_select_top_headers(
        self, all_skills_data: Dict[str, Dict[str, List[str]]],
        previous_structure: Optional[Dict[str, Dict[str, List[str]]]] = None,
        feedback: Optional[str] = None
    ) -> Dict[str, Dict[str, List[str]]]:
        """ Step 2/4: Selects/Refines top headers using LLM calls. """
        log_prefix = "Step 4 (Refine)" if feedback else "Step 2 (Select)"
        self.logger.info(f"Starting {log_prefix}: Selecting/Refining top headers. Feedback provided: {'Yes' if feedback else 'No'}")

        prompt_template = STEP4_REFINE_HEADERS_PROMPT if feedback else STEP2_SELECT_HEADERS_PROMPT
        inputs = {
            "all_skills_data_json": json.dumps(all_skills_data, indent=2)
        }
        if feedback and previous_structure:
            inputs["previous_structure_json"] = json.dumps(previous_structure, indent=2)
            inputs["feedback"] = feedback
        elif feedback and not previous_structure:
             self.logger.error("Feedback provided for refinement (Step 4) but no previous structure available.")
             return {}

        input_vars = list(inputs.keys())

        try:
            result_str = run_llm_chain(
                self.llm, prompt_template, input_vars, inputs, fail_softly=True
            )
            self.logger.debug(f"{log_prefix} LLM Raw Output:\n{result_str}")

            if result_str.startswith("[LLM FAILED]"):
                self.logger.error(f"LLM failed during {log_prefix}: {result_str}")
                return {}

            json_str = extract_json_block(result_str)
            if not json_str:
                self.logger.error(f"Could not extract JSON block from {log_prefix} LLM. Raw: {result_str}")
                return {}

            selected_headers_data = json.loads(json_str)

            # --- Validation and Recovery ---
            if not isinstance(selected_headers_data, dict):
                 self.logger.error(f"{log_prefix} LLM did not return a JSON object. Output: {json_str}")
                 return {}

            # Ensure Soft Skills is present, copying from original data if necessary
            if "Soft Skills" not in selected_headers_data:
                self.logger.warning(f"{log_prefix} LLM omitted 'Soft Skills'. Adding it back from Step 1 data.")
                if "Soft Skills" in all_skills_data:
                    selected_headers_data["Soft Skills"] = all_skills_data["Soft Skills"]
                else:
                    self.logger.error("Critical error: 'Soft Skills' not found in Step 1 data either.")
                    return {} # Cannot proceed without soft skills

            # Ensure exactly 5 headers
            if len(selected_headers_data) != 5:
                self.logger.warning(f"{log_prefix} LLM returned {len(selected_headers_data)} headers instead of 5. Attempting recovery.")
                recovered_data = {"Soft Skills": selected_headers_data["Soft Skills"]}
                tech_count = 0
                # Prioritize headers that were in the previous structure if refining
                headers_to_consider = list(previous_structure.keys()) if feedback and previous_structure else list(selected_headers_data.keys())
                for header in headers_to_consider:
                    if header != "Soft Skills" and header in selected_headers_data and tech_count < 4:
                         # Ensure the data exists in the full dataset for consistency
                         if header in all_skills_data:
                              recovered_data[header] = all_skills_data[header]
                              tech_count += 1
                         else:
                              self.logger.warning(f"Header '{header}' selected by LLM not found in original data.")

                # If still not 5, try adding from remaining selected headers
                if len(recovered_data) < 5:
                     for header in selected_headers_data:
                          if header not in recovered_data and header != "Soft Skills" and tech_count < 4:
                               if header in all_skills_data:
                                    recovered_data[header] = all_skills_data[header]
                                    tech_count += 1
                               else:
                                    self.logger.warning(f"Header '{header}' selected by LLM not found in original data.")
                          if len(recovered_data) == 5: break


                if len(recovered_data) == 5:
                    self.logger.info("Recovered 5 headers successfully.")
                    selected_headers_data = recovered_data
                else:
                    self.logger.error(f"Header recovery failed. Could only recover {len(recovered_data)} headers.")
                    return {}
            # --- End Validation and Recovery ---


            self.logger.info(f"Finished {log_prefix}: Header selection/refinement.")
            self.logger.debug(f"{log_prefix} Output: {json.dumps(selected_headers_data, indent=2)}")
            return selected_headers_data

        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse extracted JSON from {log_prefix} LLM: {json_str}")
            return {}
        except Exception as e:
            self.logger.error(f"Error in {log_prefix}: {e}", exc_info=True)
            return {}
        
    # Step functions (_step1_extract_skills_per_header, _step2_select_top_headers, etc.)
    # remain as you wrote, but you will:
    # - use JudgeEvaluation(score=..., comment=...) instead of pass_field
    # - use `get_judge_tool_schema()` from schemas/tools.py
    # - use `extract_json_block()` from utils/json_extractor.py

    # Your `_step3_judge_verification` should now accept JudgeEvaluation with score only.
    # You can update pass/fail logic using a threshold (e.g., score >= 90 is a pass).

    # Example judge response handling:
    def _step3_judge_verification(
        self, selected_skills_data: Dict[str, Dict[str, List[str]]], job_description: str
    ) -> Tuple[bool, str]:
        """Judge evaluation step using tool schema."""
        self.logger.info("Running Step 3: Judge Verification")

        if not self.supports_tool_calling:
            self.logger.warning("Tool calling not supported. Skipping judge.")
            return True, "Skipped"

        tools = get_judge_tool_schema()
        messages = [{
            "role": "user",
            "content": f"""
            Evaluate the proposed skills section below against the job description. If any improvements are needed, return a score below 90 and specify all issues.

            Job Description:
            {job_description}

            Proposed Skills Section:
            {json.dumps(selected_skills_data, indent=2)}
            """
        }]

        try:
            response = self.raw_llm_client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                messages=messages,
                tools=tools,
                tool_choice={"type": "tool", "name": "verify_skills_section"},
            )
            if response.content and hasattr(response.content[0], 'type') and response.content[0].type == 'tool_use':
                tool_input = response.content[0].input
                evaluation = JudgeEvaluation(**tool_input)
                return evaluation.score >= 90, evaluation.comment
            else:
                self.logger.error("Tool not invoked by model.")
                return False, "Tool not used"
        except Exception as e:
            self.logger.error(f"Judge evaluation failed: {e}", exc_info=True)
            return False, str(e)

    # Your refine_skills(), _step1, _step2, _format_skills_output remain the same structure.
    # You only swap: pass_field -> score, schema source -> get_judge_tool_schema
    # Prompt: clarify how score works

# Done. Let me know if you'd like me to populate __init__.py files or generate tests.
