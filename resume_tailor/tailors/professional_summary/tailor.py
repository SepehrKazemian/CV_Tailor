from typing import Dict, Optional
from resume_tailor.utils.llm_provider import get_llm
from resume_tailor.tailors.professional_summary.llm_gen import generate_summary, rewrite_summary
from resume_tailor.tailors.professional_summary.judge_llm import evaluate_summary
from resume_tailor.tailors.professional_summary.logger_config import get_logger
import resume_tailor.tailors.professional_summary.judge_llm as jl

logger = get_logger(__name__)
from importlib import reload


class ProfessionalSummaryTailor:
    """
    Generates and iteratively refines a professional summary.
    """

    MAX_REFINEMENT_ITERATIONS = 5

    def __init__(self, llm_provider: str, api_key: Optional[str] = None):
        self.model_name, self.llm, self.raw_client = get_llm(
            llm_provider, api_key, return_raw=True
        )
        self.tool_supported = (
            self.raw_client
            and hasattr(self.raw_client, "messages")
            and hasattr(self.raw_client.messages, "create")
        )

    def refine_summary(
        self, analysis: Dict, original_summary: str, job_desc: str
    ) -> str:
        """
        Iteratively refines the professional summary.

        Args:
            analysis: Analysis extracted from job description.
            original_summary: Existing professional summary.
            job_desc: Full job description text.

        Returns:
            str: Final refined summary.
        """
        reload(jl)
        current = generate_summary(self.llm, analysis, original_summary)
        print(f"current1 {current}")

        if "[LLM FAILED]" in current:
            return original_summary

        for i in range(self.MAX_REFINEMENT_ITERATIONS):
            logger.info(f"Refinement Iteration {i + 1}")

            if not self.tool_supported:
                return current

            feedback = jl.evaluate_summary(
                self.raw_client, self.model_name, current, job_desc
            )
            print(f"evaluation1 {feedback}")
            if not feedback:
                return current

            new_summary = rewrite_summary(self.llm, current, feedback)
            print(f"new_summary is {new_summary}")
            if "[LLM FAILED]" in new_summary:
                return current
            current = new_summary

        return current
