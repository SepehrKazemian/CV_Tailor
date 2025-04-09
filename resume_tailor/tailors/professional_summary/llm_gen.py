from typing import Dict, Any
from resume_tailor.utils.llm_utils import run_llm_chain
from resume_tailor.utils import prompt_templates as pt
from importlib import reload
from resume_tailor.tailors.professional_summary.logger_config import get_logger
from resume_tailor.utils.prompt_templates import REWRITE_SUMMARY_PROMPT

logger = get_logger(__name__)


def generate_summary(llm, analysis: Dict[str, Any], original_summary: str) -> str:
    """
    Generate a professional summary based on the job analysis.

    Args:
        llm: LLM instance to run the prompt.
        analysis: Job analysis dictionary.
        original_summary: Current or existing summary.

    Returns:
        str: Generated summary.
    """
    logger.info("Starting initial professional summary generation.")
    inputs = {
        "business": analysis.get("business", "N/A"),
        "responsibilities": analysis.get("responsibilities", []),
        "keywords": analysis.get("keywords", []),
        "tech_stack": analysis.get("tech_stack", []),
        "experience_focus": analysis.get("experience_focus", []),
        "domain_knowledge": analysis.get("domain_knowledge", []),
        "soft_skills": analysis.get("soft_skills", []),
        "original_summary": original_summary,
    }

    reload(pt)
    return run_llm_chain(
        llm, pt.PROFESSIONAL_SUMMARY_PROMPT, list(inputs.keys()), inputs, fail_softly=True
    )


def rewrite_summary(llm, summary: str, feedback: str) -> str:
    """
    Rewrite a summary based on feedback.

    Args:
        llm: The LLM instance.
        summary: The original summary.
        feedback: Feedback string.

    Returns:
        str: The rewritten summary.
    """
    inputs = {"original_summary": summary, "feedback": feedback}
    return run_llm_chain(
        llm, REWRITE_SUMMARY_PROMPT, list(inputs.keys()), inputs, fail_softly=True
    )
