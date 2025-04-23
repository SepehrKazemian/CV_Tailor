import logging
import json
import re
import ast # For literal_eval
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field, ValidationError

from resume_tailor.utils.llm_provider import get_llm
from resume_tailor.utils.llm_utils import run_llm_chain
# Import necessary prompt templates (will define these later)
# from resume_tailor.utils.prompt_templates import (
#     EXP_STEP1_TAILOR_BULLETS_PROMPT,
#     EXP_STEP2_JUDGE_PROMPT, # Implicitly via tool call
#     EXP_STEP2_RETRY_PROMPT # Implicitly via tool call feedback
# )
from resume_tailor.utils.file_utils import format_experience_output # For final formatting
# Import utilities from tailor_utils
from resume_tailor.tailors.tailor_utils import extract_json_block

# Configure logging for this module
log_file = "experience_refinement.log"
try:
    with open(log_file, "w"): pass
except OSError as e:
    print(f"Warning: Could not clear log file {log_file}: {e}")

# Use root logger for setup warnings
root_logger_for_setup = logging.getLogger(__name__ + "_setup")
if not root_logger_for_setup.handlers:
     handler = logging.StreamHandler() # Log setup issues to console
     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
     handler.setFormatter(formatter)
     root_logger_for_setup.addHandler(handler)
     root_logger_for_setup.setLevel(logging.INFO)

# --- Pydantic Models for Judge Tool ---
class ExperienceJudgeEvaluation(BaseModel):
    pass_field: bool = Field(..., alias="pass")
    comment: str = Field(
        default="",
        description="Feedback if validation fails. If pass=false, THIS MUST CONTAIN the corrected list of bullet points."
    )

# --- Main Class ---
class ExperienceRefiner:
    """
    Handles the tailoring and validation of the experience section using YAML structure.
    """
    MAX_REFINEMENT_ITERATIONS = 5 # Max retries for the judge validation loop
    log_file = log_file # Make log file path accessible

    def __init__(self, llm_provider: str, api_key: Optional[str] = None):
        """ Initializes the ExperienceRefiner. """
        self.model_name, self.llm, self.raw_llm_client = get_llm(
            llm_provider, api_key, return_raw=True
        )
        # Use instance logger from now on
        self._configure_logger() # Call logger configuration
        self.logger.info(f"Initialized ExperienceRefiner with model: {self.model_name}")

        self.supports_tool_calling = (
            self.raw_llm_client and hasattr(self.raw_llm_client, 'messages') and
            hasattr(self.raw_llm_client.messages, 'create')
        )
        if not self.supports_tool_calling:
             self.logger.warning("Raw LLM client does not support tool calling. Judge step might fail.")

    # Added logger configuration method
    def _configure_logger(self):
        """Sets up file logging for this instance."""
        self.logger = logging.getLogger(f"{__name__}.{id(self)}") # Instance-specific logger
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False # Prevent duplicate logs in root logger
        # Remove existing handlers to avoid duplication if re-initialized
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()
        # Add file handler
        try:
            with open(self.log_file, "w"): pass # Clear log file on init
            fh = logging.FileHandler(self.log_file)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        except OSError as e:
            # Use root logger for this warning as instance logger might not be set up
            root_logger_for_setup.warning(f"Could not configure file logging for ExperienceRefiner: {e}")


    def _get_judge_tool_schema(self) -> List[Dict[str, Any]]:
        """Returns the schema for the experience judge tool."""
        schema = ExperienceJudgeEvaluation.model_json_schema()
        # Adjust schema based on Pydantic alias 'pass' for 'pass_field'
        if 'properties' in schema and 'pass_field' in schema['properties']:
            schema['properties']['pass'] = schema['properties'].pop('pass_field')
            if 'pass_field' in schema.get('required', []):
                schema['required'].remove('pass_field')
                if 'pass' not in schema['required']:
                    schema['required'].append('pass')
        return [{
            "name": "validate_experience_bullets",
            "description": "Validates rewritten experience bullet points for relevance, conciseness, impact, and non-repetition.",
            "input_schema": schema
        }]

    def _step1_tailor_bullets(
        self, original_bullets: List[str], job_description: str, analysis: Dict[str, Any]
    ) -> List[str]:
        """ Step 1: Tailor a list of bullet points using an LLM. """
        self.logger.info(f"Starting Step 1: Tailoring {len(original_bullets)} bullet points.")
        # Define prompt inline for now
        EXP_STEP1_TAILOR_BULLETS_PROMPT = """
        You are an expert resume writer tailoring experience bullet points for a specific job.
        Job Description:
        {job_description}

        Job Analysis Insights:
        - Keywords: {keywords}
        - Tech Stack: {tech_stack}
        - Experience Focus: {experience_focus}
        - Tone: {tone}

        Original Bullet Points for this specific project/job:
        {original_bullets_str}

        Instructions:
        1. Select only the bullet points most relevant to the job description and analysis insights provided above. Discard irrelevant points entirely.
        2. Rewrite the selected bullet points to:
           - Align closely with the job requirements and keywords. Use terminology from the job description where appropriate.
           - Be concise and impactful (strictly maximum 2 lines each). Start with strong action verbs.
           - Quantify achievements with metrics or specific outcomes whenever possible.
           - Use a {tone} tone (e.g., technical, innovative, business-focused).
           - Substitute technologies mentioned in the original bullet points with equivalent technologies from the job's tech stack ({tech_stack}) ONLY if appropriate and the accomplishment is transferable (e.g., replace GCP with AWS if the job requires AWS and the task is cloud-agnostic). Be cautious with substitutions; only do it if it makes sense.
        3. Output ONLY a Python-style list of strings, where each string is a rewritten bullet point.
           Example: ["Rewritten bullet 1.", "Rewritten bullet 2."]
           Return an empty list `[]` if no original bullet points are relevant.
        """
        inputs = {
            "job_description": job_description,
            "keywords": analysis.get("keywords", []),
            "tech_stack": analysis.get("tech_stack", []),
            "experience_focus": analysis.get("experience_focus", []),
            "tone": analysis.get("tone", "technical"),
            "original_bullets_str": "\n".join([f"- {b}" for b in original_bullets])
        }
        input_vars = list(inputs.keys())

        result_str = run_llm_chain(
            self.llm, EXP_STEP1_TAILOR_BULLETS_PROMPT, input_vars, inputs, fail_softly=True
        )
        self.logger.debug(f"Step 1 LLM Raw Output:\n{result_str}")

        if result_str.startswith("[LLM FAILED]"):
            self.logger.warning(f"LLM failed during bullet point tailoring: {result_str}. Returning original bullets.")
            return original_bullets

        # Attempt to parse the list output robustly
        rewritten_bullets = []
        try:
            clean_result = result_str.strip()
            # Remove potential markdown fences first
            if clean_result.startswith("```python"):
                clean_result = clean_result[9:]
            elif clean_result.startswith("```"):
                 clean_result = clean_result[3:]
            if clean_result.endswith("```"):
                clean_result = clean_result[:-3]
            clean_result = clean_result.strip()

            # Handle potential Python list literal string
            if clean_result.startswith('[') and clean_result.endswith(']'):
                 parsed_list = ast.literal_eval(clean_result)
                 if isinstance(parsed_list, list):
                      rewritten_bullets = [str(b).strip() for b in parsed_list]
                 else:
                      raise ValueError("Parsed result is not a list.")
            else:
                 # Assume newline-separated bullets if not a list literal
                 rewritten_bullets = [b.strip("- ").strip() for b in clean_result.split('\n') if b.strip()]

            self.logger.info(f"Successfully tailored {len(original_bullets)} bullets into {len(rewritten_bullets)}.")
            return rewritten_bullets

        except (ValueError, SyntaxError, TypeError) as e:
            self.logger.warning(f"Failed to parse Step 1 LLM output into list: {e}. Raw: {result_str}. Returning original.")
            return original_bullets
        except Exception as e:
             self.logger.error(f"Unexpected error parsing Step 1 LLM output: {e}", exc_info=True)
             return original_bullets


    def _step2_validate_bullets(
        self,
        rewritten_bullets: List[str],
        job_description: str,
        all_previous_bullets: List[str]
    ) -> List[str]:
        """ Step 2: Validate and de-duplicate bullets using LLM Judge with retry loop. """
        self.logger.info(f"Starting Step 2: Validating {len(rewritten_bullets)} rewritten bullet points.")
        if not self.supports_tool_calling:
            self.logger.warning("Skipping validation/deduplication - tool calling not supported.")
            return rewritten_bullets

        current_bullets_to_validate = rewritten_bullets
        tools = self._get_judge_tool_schema()

        for attempt in range(self.MAX_REFINEMENT_ITERATIONS):
            self.logger.info(f"Judge Validation Attempt {attempt + 1}/{self.MAX_REFINEMENT_ITERATIONS}")
            bullets_str = "\n".join([f"- {b}" for b in current_bullets_to_validate])
            previous_bullets_str = "\n".join([f"- {b}" for b in all_previous_bullets]) if all_previous_bullets else "None"

            prompt_content = f"""
            You are an expert resume reviewer acting as a judge. Validate the following rewritten experience bullet points based on the job description and previously validated bullets.

            Job Description:
            {job_description}

            Bullet Points to Validate:
            {bullets_str}

            Previously Validated Bullet Points (for checking repetition):
            {previous_bullets_str}

            Validation Criteria:
            1. Accuracy: Do the bullet points accurately reflect relevant experience for the job description?
            2. Conciseness & Impact: Is the language concise (max 2 lines per bullet), impactful, and results-oriented?
            3. Repetition: Do these bullet points significantly repeat skills/tools/achievements already covered in the 'Previously Validated Bullet Points'? Minor overlap is okay if context differs, but avoid redundancy.

            Instructions:
            Use the 'validate_experience_bullets' tool.
            - If ALL criteria are met, set 'pass' to true and leave 'comment' empty or provide brief positive feedback.
            - If ANY criterion fails, set 'pass' to false. The 'comment' field MUST contain the REVISED list of bullet points, corrected to address the failures (e.g., remove repetitive points, reword for clarity/impact, ensure relevance). Ensure the comment ONLY contains the bullet points, formatted as a simple list separated by newlines (e.g., "Corrected bullet 1.\\nCorrected bullet 2.").
            """
            messages = [{"role": "user", "content": prompt_content}]

            try:
                response = self.raw_llm_client.messages.create(
                    model=self.model_name,
                    max_tokens=1024,
                    messages=messages,
                    tools=tools,
                    tool_choice={"type": "tool", "name": "validate_experience_bullets"},
                )
                self.logger.debug(f"Raw judge response object (Attempt {attempt+1}): {response}")

                tool_input = None
                if response.content and hasattr(response.content[0], 'type') and response.content[0].type == 'tool_use':
                    tool_input = response.content[0].input
                    self.logger.info(f"Raw judge tool input received (Attempt {attempt+1}): {tool_input}")
                else:
                    self.logger.error(f"No valid tool use found in judge response (Attempt {attempt+1}): {response.content}")
                    if attempt == self.MAX_REFINEMENT_ITERATIONS - 1:
                         self.logger.warning("Judge failed to use tool on final attempt. Returning last valid bullets.")
                         return current_bullets_to_validate
                    else:
                         self.logger.warning("Judge failed to use tool. Retrying validation.")
                         continue

                evaluation = ExperienceJudgeEvaluation(**tool_input)
                self.logger.info(f"Parsed judge evaluation (Attempt {attempt+1}): Pass={evaluation.pass_field}")

                if evaluation.pass_field:
                    self.logger.info(f"Validation passed on attempt {attempt + 1}.")
                    return current_bullets_to_validate
                else:
                    self.logger.info(f"Validation failed. Feedback: {evaluation.comment}")
                    # Expecting comment to be newline-separated list of corrected bullets
                    corrected_bullets = [b.strip("- ").strip() for b in evaluation.comment.split('\n') if b.strip()]
                    if not corrected_bullets:
                         self.logger.warning("Judge failed but comment did not contain corrected bullets. Retrying with original failed bullets.")
                    else:
                         self.logger.info(f"Using corrected bullets from judge comment for next attempt: {corrected_bullets}")
                         current_bullets_to_validate = corrected_bullets

                    if attempt == self.MAX_REFINEMENT_ITERATIONS - 1:
                        self.logger.warning("Max validation retries reached. Returning last set of bullets provided by judge.")
                        return current_bullets_to_validate

            except ValidationError as e:
                self.logger.error(f"Pydantic validation error parsing judge response (Attempt {attempt+1}): {e}", exc_info=True)
                self.logger.error(f"Invalid judge tool input data received: {tool_input}")
                if attempt == self.MAX_REFINEMENT_ITERATIONS - 1: return current_bullets_to_validate
            except Exception as e:
                self.logger.error(f"Error during judge verification (Attempt {attempt+1}): {e}", exc_info=True)
                if attempt == self.MAX_REFINEMENT_ITERATIONS - 1: return current_bullets_to_validate

        self.logger.error("Exited validation loop unexpectedly.")
        return rewritten_bullets # Fallback

    def refine_experience(
        self,
        original_experience_data: Dict[str, Any],
        job_description: str,
        analysis: Dict[str, Any]
    ) -> str:
        """
        Orchestrates the tailoring and validation process for the entire experience section.

        Args:
            original_experience_data: Parsed dictionary from experience.yml.
            job_description: The full job description text.
            analysis: The job analysis dictionary.

        Returns:
            The final refined experience section as a formatted string.
        """
        self.logger.info("=== Starting Experience Section Refinement ===")
        all_validated_bullets = []
        final_experience_structure = {}

        for job_key, job_content in original_experience_data.items():
            self.logger.info(f"Processing Job: {job_key}")
            final_experience_structure[job_key] = {}

            if isinstance(job_content, list):
                task_key = f"{job_key}_direct"
                self.logger.info(f"Processing Task: {task_key} (Direct Bullets)")
                original_bullets = job_content
                rewritten_bullets = self._step1_tailor_bullets(original_bullets, job_description, analysis)
                validated_bullets = self._step2_validate_bullets(rewritten_bullets, job_description, all_validated_bullets)
                final_experience_structure[job_key] = validated_bullets
                all_validated_bullets.extend(validated_bullets)
                self.logger.info(f"Finished Task: {task_key}. Added {len(validated_bullets)} bullets.")

            elif isinstance(job_content, dict):
                for project_key, project_bullets in job_content.items():
                    if isinstance(project_bullets, list):
                        task_key = f"{job_key}.{project_key}"
                        self.logger.info(f"Processing Task: {task_key}")
                        original_bullets = project_bullets
                        rewritten_bullets = self._step1_tailor_bullets(original_bullets, job_description, analysis)
                        validated_bullets = self._step2_validate_bullets(rewritten_bullets, job_description, all_validated_bullets)
                        # Ensure the job key exists before assigning project key
                        if job_key not in final_experience_structure:
                             final_experience_structure[job_key] = {}
                        final_experience_structure[job_key][project_key] = validated_bullets
                        all_validated_bullets.extend(validated_bullets)
                        self.logger.info(f"Finished Task: {task_key}. Added {len(validated_bullets)} bullets.")
                    else:
                        self.logger.warning(f"Skipping invalid content under project '{project_key}' for job '{job_key}'. Expected a list.")
            else:
                 self.logger.warning(f"Skipping invalid content under job '{job_key}'. Expected list or dict.")

        self.logger.info("=== Finished Experience Section Refinement ===")
        return format_experience_output(final_experience_structure)
