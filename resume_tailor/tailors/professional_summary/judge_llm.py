from typing import List, Dict, Any, Optional
from pydantic import ValidationError
from resume_tailor.tailors.professional_summary.model_fields import (
    ProfessionalSummaryEvaluation,
)
from resume_tailor.utils.pydantic_to_schema import pydantic_model_to_schema
from resume_tailor.tailors.professional_summary.logger_config import get_logger
from resume_tailor.tailors.professional_summary import model_fields as mf
from importlib import reload

logger = get_logger(__name__)


def get_tool_schema() -> List[Dict[str, Any]]:
    """Returns the Claude-compatible tool schema."""
    return [
        {
            "name": "evaluate_professional_summary",
            "description": "Evaluates whether a professional summary meets key criteria.",
            "input_schema": pydantic_model_to_schema(mf.ProfessionalSummaryEvaluation),
        }
    ]


def evaluate_summary(
    client, model_name: str, summary: str, job_desc: str
) -> Optional[ProfessionalSummaryEvaluation]:
    """
    Evaluate summary against job description using LLM tool-calling.

    Args:
        client: Raw LLM client.
        model_name: Name of the LLM model.
        summary: Generated professional summary.
        job_desc: Original job description.

    Returns:
        ProfessionalSummaryEvaluation: Structured evaluation object or None on failure.
    """
    reload(mf)
    logger.info(f"Evaluating summary with model: {model_name}")
    tools = get_tool_schema()
    messages = [
        {
            "role": "user",
            "content": (
                "You are an expert resume reviewer. Evaluate the following professional summary "
                "as if you're critiquing it for a high-level job application. Be critical but constructive.\n\n"
                "Evaluation Instructions:"
                "For 'length', return a 'pass' boolean:"
                "- Pass if under ~100 words or 5 lines and clearly written."

                "For 'content alignment' and 'best practices', return:"
                "- A 'score' (0–100, integer):"
                "    • 90–100 = excellent; no meaningful revisions needed"
                "    • 70–89 = strong but improvable"
                "    • 40–69 = moderate content or tone gaps"
                "    • below 40 = major issues"
                "- A 'comment' with actionable feedback."
                "- Only deduct points if your suggestion would truly improve clarity, impact, or relevance."
                "- Do not penalize for style preferences or optional rewording."
                "Avoid using numbers, metrics, or buzzwords for improvements. Keep summaries clean and executive-friendly."
                f"Professional Summary:\n{summary}\n\n"
                f"Job Description:\n{job_desc}"
            )
        }
    ]

    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=1024,
            messages=messages,
            tools=tools,
            tool_choice={"type": "tool", "name": "evaluate_professional_summary"},
        )

        tool_input = None
        if (
            response.content
            and hasattr(response.content[0], "type")
            and response.content[0].type == "tool_use"
        ):
            tool_input = response.content[0].input
        else:
            logger.error("Tool use response not found.")
            return None

        evaluation = mf.ProfessionalSummaryEvaluation(**tool_input)
        print(evaluation)

        SCORE_THRESHOLD = {
            "content_alignment": 90,
            "best_practices": 90,
        }
        if (
            evaluation.length_check.pass_field
            and evaluation.content_alignment.score >= SCORE_THRESHOLD["content_alignment"]
            and evaluation.best_practices_check.score >= SCORE_THRESHOLD["best_practices"]
        ):
            return ""  # No rewrite needed

        # ✅ Feedback aggregation based on score failures or length fail
        feedback = "\n".join(
            filter(
                None,
                [
                    (
                        f"- Length: {evaluation.length_check.comment}"
                        if not evaluation.length_check.pass_field
                        else ""
                    ),
                    (
                        f"- Content Alignment (score {evaluation.content_alignment.score}/100): {evaluation.content_alignment.comment}"
                        if evaluation.content_alignment.score < SCORE_THRESHOLD["content_alignment"]
                        else ""
                    ),
                    (
                        f"- Best Practices (score {evaluation.best_practices_check.score}/100): {evaluation.best_practices_check.comment}"
                        if evaluation.best_practices_check.score < SCORE_THRESHOLD["best_practices"]
                        else ""
                    ),
                ]
            )
        )
        return feedback
    except ValidationError as ve:
        logger.error(f"Validation error during evaluation parsing: {ve}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        return None
