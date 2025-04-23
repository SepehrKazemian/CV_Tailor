from typing import List, Dict, Optional
import json
from resume_tailor.utils.llm_utils import run_llm_chain
from resume_tailor.utils import prompt_templates as pt
# import (
#     STEP1_EXTRACT_SKILLS_PROMPT,
#     STEP2_SELECT_HEADERS_PROMPT,
#     STEP4_REFINE_HEADERS_PROMPT
# )
from importlib import reload
import re

class SkillHeaderSelector:
    """Encapsulates logic for selecting and refining skill section headers using LLM."""

    def __init__(self, llm, logger):
        self.llm = llm
        self.logger = logger

    def select_top_headers(
        self,
        all_skills_data: Dict[str, Dict[str, List[str]]],
        previous_structure: Optional[Dict[str, Dict[str, List[str]]]] = None,
        feedback: Optional[str] = None
    ) -> Dict[str, Dict[str, List[str]]]:
        """Selects or refines top headers from all extracted skills using the LLM."""
        reload(pt)
        log_prefix = "Step 4 (Refine)" if feedback else "Step 2 (Select)"
        self.logger.info(f"Starting {log_prefix}: Feedback provided: {'Yes' if feedback else 'No'}")

        prompt_template = pt.STEP4_REFINE_HEADERS_PROMPT if feedback else pt.STEP2_SELECT_HEADERS_PROMPT
        inputs = {"all_skills_data_json": json.dumps(all_skills_data)}

        if feedback:
            if not previous_structure:
                self.logger.error("Feedback provided but no previous structure for refinement. Aborting.")
                return {}
            inputs["previous_structure_json"] = json.dumps(previous_structure, indent=2)
            inputs["feedback"] = feedback

        input_vars = list(inputs.keys())

        try:
            from resume_tailor.utils import llm_utils as lu
            result_str = run_llm_chain(
                self.llm, prompt_template, input_vars, inputs, fail_softly=True
            )
            self.logger.debug(f"{log_prefix} LLM Raw Output:\n{result_str}")

            if result_str.startswith("[LLM FAILED]"):
                self.logger.error(f"LLM failed during {log_prefix}: {result_str}")
                return {}

            json_str = self.parse_selected_skills_output(result_str)
            if not json_str:
                self.logger.error(f"Could not extract JSON block from LLM output: {result_str}")
                return {}

            if not isinstance(json_str, dict):
                self.logger.error(f"Expected dict from LLM, got: {type(json_str)}")
                return {}

            return self._validate_selected_headers(json_str, all_skills_data, previous_structure, feedback)

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON in {log_prefix}: {e}\nRaw: {json_str}")
            return {}
        except Exception as e:
            self.logger.error(f"Unexpected error in {log_prefix}: {e}", exc_info=True)
            return {}

    def _validate_selected_headers(
        self,
        selected_headers: Dict[str, Dict[str, List[str]]],
        all_skills_data: Dict[str, Dict[str, List[str]]],
        previous_structure: Optional[Dict[str, Dict[str, List[str]]]] = None,
        feedback: Optional[str] = None
    ) -> Dict[str, Dict[str, List[str]]]: 
        """Validates and recovers the selected headers to ensure 'Soft Skills' is present and total is 5."""

        if "Soft Skills" not in selected_headers:
            self.logger.warning("'Soft Skills' missing from selected headers. Attempting recovery.")
            if "Soft Skills" in all_skills_data:
                selected_headers["Soft Skills"] = all_skills_data["Soft Skills"]
            else:
                self.logger.error("'Soft Skills' not found in Step 1 data. Cannot proceed.")
                return {}

        if len(selected_headers) == 5:
            self.logger.info("Correct number of headers selected. Validation passed.")
            return selected_headers

        self.logger.warning(f"Expected 5 headers, got {len(selected_headers)}. Attempting recovery.")
        recovered = {"Soft Skills": selected_headers["Soft Skills"]}
        tech_count = 0

        preferred_order = list(previous_structure.keys()) if feedback and previous_structure else list(selected_headers.keys())
        for h in preferred_order:
            if h != "Soft Skills" and h in selected_headers and tech_count < 4:
                if h in all_skills_data:
                    recovered[h] = all_skills_data[h]
                    tech_count += 1
                else:
                    self.logger.warning(f"Header '{h}' not found in Step 1 data.")

        if len(recovered) < 5:
            for h in selected_headers:
                if h not in recovered and h != "Soft Skills" and tech_count < 4:
                    if h in all_skills_data:
                        recovered[h] = all_skills_data[h]
                        tech_count += 1
                if len(recovered) == 5:
                    break

        if len(recovered) == 5:
            self.logger.info("Header count recovered successfully.")
            return recovered

        self.logger.error(f"Recovery failed. Only {len(recovered)} headers available.")
        return {}
    
    
    def extract_json_block(self, text: str) -> str:
        """
        Extracts the first valid JSON object (between { and }) from any text.

        Strips out:
        - Markdown fences like ```json or ```
        - Any text before the first opening '{'
        
        Raises:
            ValueError: If no valid JSON object is found.
        """
        text = text.strip()

        # Remove markdown code fences
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]

        # Remove any text before first {
        first_curly = text.find('{')
        if first_curly != -1:
            text = text[first_curly:]

        # Find and return first full JSON block
        match = re.search(r'\{[\s\S]+?\}', text)
        if match:
            return match.group(0)

        raise ValueError("No valid JSON block found in LLM output.")


    def parse_selected_skills_output(self, raw_text: str) -> Dict[str, List[str]]:
        """
        Parses flat JSON structure from LLM with headers mapped to a list of selected skills.
        
        Example expected input:
        {
        "LLM Ecosystem & GenAI": ["LangChain", "LlamaIndex"],
        "Soft Skills": ["Communication", "Leadership"]
        }

        Returns:
            Dict[str, List[str]]: Clean mapping of header to skill list.
        """
        json_str = self.extract_json_block(raw_text)
        try:
            parsed = json.loads(json_str)
            return parsed
        except:
            raise ValueError(f"Output cannot be parsed, please retry!")