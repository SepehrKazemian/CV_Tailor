from typing import List, Dict, Any, Tuple
import json
import logging

from resume_tailor.utils.pydantic_to_schema import pydantic_model_to_schema
from resume_tailor.models.judge import JudgeEvaluation  # your pydantic model
from resume_tailor.schema.skills_judge_schema import get_judge_tool_schema  # your tool schema generator


class SkillsJudgeVerifier:
    """
    Judge evaluator for verifying selected skills headers using tool-calling LLMs.
    """

    def __init__(self, llm_raw, model_name: str, logger: logging.Logger):
        """
        Args:
            llm_raw: Raw LLM client that supports tool-calling (e.g., Claude client).
            model_name: Name of the model to use.
            logger: Logger object for logging outputs.
        """
        self.llm_raw = llm_raw
        self.model_name = model_name
        self.logger = logger

    def run_judge(
        self,
        selected_skills_data: Dict[str, Dict[str, List[str]]],
        job_description: str
    ) -> Tuple[bool, str]:
        """
        Run the judge verification tool on the selected skills.

        Args:
            selected_skills_data: The selected 5-skill-section output from the skill extractor.
            job_description: Full job description.

        Returns:
            Tuple[bool, str]: (True if passed, comment with feedback if failed or skipped)
        """
        self.logger.info("Running Step 3: Judge Verification")

        tools = get_judge_tool_schema()
        messages = self._build_judge_messages(selected_skills_data, job_description)

        try:
            response = self._call_llm_tool(messages, tools)
            return self._parse_response(response)
        except Exception as e:
            self.logger.error(f"Judge evaluation failed: {e}", exc_info=True)
            return False, str(e)

    def _build_judge_messages(
        self,
        selected_skills_data: Dict[str, Dict[str, List[str]]],
        job_description: str
    ) -> List[Dict[str, str]]:
        """
        Constructs the user message for LLM tool input.

        Returns:
            List of formatted message dicts.
        """
        return [{
            "role": "user",
            "content": (
                "Evaluate the proposed skills section below against the job description. "
                "If any improvements are needed, return a score below 90 and specify all issues.\n\n"
                f"Job Description:\n{job_description}\n\n"
                f"Proposed Skills Section:\n{json.dumps(selected_skills_data, indent=2)}"
            )
        }]

    def _call_llm_tool(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]]
    ) -> Any:
        """
        Calls the LLM with provided messages and tools.

        Returns:
            The LLM response.
        """
        return self.llm_raw.messages.create(
            model=self.model_name,
            max_tokens=1024,
            messages=messages,
            tools=tools,
            tool_choice={"type": "tool", "name": "verify_skills_section"},
        )

    def _parse_response(self, response: Any) -> Tuple[bool, str]:
        """
        Parses the response from the judge tool call.

        Returns:
            Tuple[bool, str]: (True if score >= 90, feedback comment)
        """
        if (
            response.content and
            hasattr(response.content[0], "type") and
            response.content[0].type == "tool_use"
        ):
            tool_input = response.content[0].input
            evaluation = JudgeEvaluation(**tool_input)
            return evaluation.score >= 90, evaluation.comment

        self.logger.error("Tool was not invoked in LLM response.")
        return False, "Tool not used"
