import logging
import json
import re
from typing import Dict, Any, Optional, Tuple, List
from pydantic import BaseModel, Field, ValidationError

from resume_tailor.utils.llm_provider import get_llm
from resume_tailor.utils.llm_utils import run_llm_chain
from resume_tailor.tailors.tailor_utils import SKILL_SECTION_HEADERS
# Import necessary prompt templates
from resume_tailor.utils.prompt_templates import (
    STEP1_EXTRACT_SKILLS_PROMPT,
    STEP2_SELECT_HEADERS_PROMPT,
    STEP4_REFINE_HEADERS_PROMPT
)

logger = logging.getLogger(__name__)
# Configure root logger if not already configured
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO)

# --- Utility Function ---
def extract_json_block(text: str) -> Optional[str]:
    """Extracts the first JSON block (between { and }) from a string."""
    # Look for JSON starting with { and ending with }
    # Handle potential markdown fences ```json ... ```
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    logger.warning("Could not find JSON block in text.")
    return None

# --- Pydantic Models ---
class JudgeEvaluation(BaseModel):
    pass_field: bool = Field(..., alias="pass")
    comment: str = Field(default="") # Comment only needed if pass is false


# --- Main Class ---
class SkillsRefiner:
    """
    Handles the iterative refinement of the skills section based on job description.
    """
    MAX_REFINEMENT_ITERATIONS = 5
    log_file = "skills_refinement.log" # Define log file name

    def __init__(self, llm_provider: str, api_key: Optional[str] = None):
        """ Initializes the SkillsRefiner. """
        self.model_name, self.llm, self.raw_llm_client = get_llm(
            llm_provider, api_key, return_raw=True
        )
        logger.info(f"Initialized SkillsRefiner with model: {self.model_name}")
        self.supports_tool_calling = (
            self.raw_llm_client and hasattr(self.raw_llm_client, 'messages') and
            hasattr(self.raw_llm_client.messages, 'create')
        )
        if not self.supports_tool_calling:
             logger.warning("Raw LLM client does not support tool calling. Judge step might fail.")
        self._configure_logger()


    def _configure_logger(self):
        """Sets up file logging for this instance."""
        self.logger = logging.getLogger(f"{__name__}.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()
        try:
            with open(self.log_file, "w"): pass
            fh = logging.FileHandler(self.log_file)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        except OSError as e:
            print(f"Warning: Could not configure file logging for SkillsRefiner: {e}")
        

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

    def _get_judge_tool_schema(self) -> List[Dict[str, Any]]:
        """Returns the schema for the judge tool."""
        schema = JudgeEvaluation.model_json_schema()
        # Adjust schema based on Pydantic alias 'pass' for 'pass_field'
        if 'properties' in schema and 'pass_field' in schema['properties']:
            schema['properties']['pass'] = schema['properties'].pop('pass_field')
            if 'pass_field' in schema.get('required', []):
                schema['required'].remove('pass_field')
                if 'pass' not in schema['required']:
                    schema['required'].append('pass')
        return [{
            "name": "verify_skills_section",
            "description": "Evaluates the selected skills section against the job description.",
            "input_schema": schema
        }]

    def _step3_judge_verification(
        self, selected_skills_data: Dict[str, Dict[str, List[str]]], job_description: str
    ) -> Tuple[bool, str]:
        """ Step 3: Judge LLM verifies the selected skills structure using tool call. """
        self.logger.info("Starting Step 3: Judge verification.")
        if not self.supports_tool_calling:
            self.logger.warning("Skipping judge verification - tool calling not supported.")
            return True, "Skipped - Tool calling not supported"

        tools = self._get_judge_tool_schema()
        selected_skills_json = json.dumps(selected_skills_data, indent=2)
        prompt_content = f"""
        You are an expert resume reviewer acting as a judge. Evaluate the following proposed skills section structure based on the full job description.

        Job Description:
        {job_description}

        Proposed Skills Section (5 headers with job/candidate skills):
        {selected_skills_json}

        Evaluation Criteria:
        - Are any important skills required by the job description missing from all lists?
        - Are any listed candidate skills clearly irrelevant to the job or the header they are under?
        - Are skills grouped under the most appropriate header?
        - Are the skill lists concise and well-structured (avoiding excessive wordiness)?
        - Are the 4 technical headers selected the most relevant ones for this specific job? Could a better header from the full list be swapped in?
        - Is the "Soft Skills" section present? (This is mandatory)

        Use the 'verify_skills_section' tool to provide your evaluation. If all criteria are met satisfactorily, set 'pass' to true. If there are issues, set 'pass' to false and provide a concise 'comment' explaining ALL the necessary corrections or improvements. Focus on actionable feedback.
        """
        messages = [{"role": "user", "content": prompt_content}]

        try:
            self.logger.info("Judge LLM call initiated.")
            response = self.raw_llm_client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                messages=messages,
                tools=tools,
                tool_choice={"type": "tool", "name": "verify_skills_section"},
            )
            self.logger.debug(f"Raw judge response object: {response}")

            tool_input = None
            if response.content and hasattr(response.content[0], 'type') and response.content[0].type == 'tool_use':
                tool_input = response.content[0].input
                self.logger.info(f"Raw judge tool input received: {tool_input}")
            else:
                self.logger.error(f"No valid tool use found in judge response: {response.content}")
                return False, "Judge LLM failed to use the required tool."

            evaluation = JudgeEvaluation(**tool_input)
            self.logger.info(f"Parsed judge evaluation: Pass={evaluation.pass_field}, Comment='{evaluation.comment}'")
            return evaluation.pass_field, evaluation.comment if not evaluation.pass_field else ""

        except ValidationError as e:
            self.logger.error(f"Pydantic validation error parsing judge response: {e}", exc_info=True)
            self.logger.error(f"Invalid judge tool input data received: {tool_input}")
            return False, f"Failed to parse judge response: {e}"
        except Exception as e:
            self.logger.error(f"Error during judge verification call or parsing: {e}", exc_info=True)
            return False, f"Error during judge verification: {e}"

    def _format_skills_output(self, final_skills_data: Dict[str, Dict[str, List[str]]]) -> str:
        """Formats the final 5-header skills data into a string for the resume."""
        output_lines = []
        header_order = [h for h in final_skills_data if h != "Soft Skills"]
        if "Soft Skills" in final_skills_data:
             header_order.append("Soft Skills")

        for header in header_order:
            if header in final_skills_data:
                skills_info = final_skills_data[header]
                combined_skills = []
                seen_skills = set()
                # Prioritize candidate skills, then add missing job skills
                for skill in skills_info.get("candidate_skills", []):
                    skill_lower = skill.strip().lower()
                    if skill_lower and skill_lower not in seen_skills:
                        combined_skills.append(skill.strip())
                        seen_skills.add(skill_lower)
                for skill in skills_info.get("job_skills", []):
                    skill_lower = skill.strip().lower()
                    if skill_lower and skill_lower not in seen_skills:
                        combined_skills.append(skill.strip())
                        seen_skills.add(skill_lower)

                if combined_skills:
                    output_lines.append(f"• {header}: {', '.join(combined_skills)}")

        return "\n".join(output_lines)

    def refine_skills(self, job_description: str, original_skills_text: str) -> str:
        """ Orchestrates the 4-step iterative refinement process for the skills section. """
        self.logger.info("=== Starting Skills Section Refinement ===")

        all_skills_data = self._step1_extract_skills_per_header(job_description, original_skills_text)
        if not all_skills_data:
            self.logger.error("Step 1 failed. Returning original skills.")
            return original_skills_text

        current_structure = None
        feedback = None
        last_successful_structure = None # Keep track of the last valid structure

        for i in range(self.MAX_REFINEMENT_ITERATIONS):
            self.logger.info(f"--- Skills Refinement Iteration {i + 1} ---")

            # Step 2 / Step 4: Select or Refine Headers
            previous_structure_for_refinement = current_structure if current_structure else all_skills_data
            if i == 0:
                current_structure = self._step2_select_top_headers(all_skills_data)
            else:
                current_structure = self._step2_select_top_headers(
                    all_skills_data, previous_structure_for_refinement, feedback
                )

            if not current_structure:
                self.logger.error(f"Step 2/4 failed in iteration {i+1}. Aborting refinement.")
                # Fallback: format the last known good structure
                fallback_structure = previous_structure_for_refinement if previous_structure_for_refinement else all_skills_data
                return self._format_skills_output(fallback_structure)

            # Check mandatory Soft Skills again after potential recovery in step 2/4
            if "Soft Skills" not in current_structure:
                self.logger.critical(f"FATAL: Soft Skills section missing after Step 2/4 recovery in iteration {i+1}. Aborting.")
                return original_skills_text

            last_successful_structure = current_structure # Store this valid structure

            # Step 3: Judge Verification
            passed, feedback = self._step3_judge_verification(current_structure, job_description)

            if passed:
                self.logger.info(f"Judge passed in iteration {i + 1}. Final structure achieved.")
                return self._format_skills_output(current_structure)

            if i < self.MAX_REFINEMENT_ITERATIONS - 1:
                self.logger.info(f"Judge failed in iteration {i + 1}. Feedback:\n{feedback}")
            else:
                self.logger.warning(f"Judge failed on final iteration {i + 1}. Using last successful structure.")
                self.logger.info(f"Final Feedback from Judge:\n{feedback}")

        self.logger.warning(f"Max refinement iterations ({self.MAX_REFINEMENT_ITERATIONS}) reached for skills.")
        # Return the last structure that was successfully generated and validated (even if judge failed last time)
        return self._format_skills_output(last_successful_structure if last_successful_structure else all_skills_data)
