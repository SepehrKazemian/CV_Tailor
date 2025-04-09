import logging
from typing import Dict, List, Any, Optional
# import json # Removed unused import

from pydantic import BaseModel, Field, ValidationError
from resume_tailor.utils.llm_provider import get_llm
from resume_tailor.utils.prompt_templates import PROFESSIONAL_SUMMARY_PROMPT, REWRITE_SUMMARY_PROMPT
from resume_tailor.utils.llm_utils import run_llm_chain

# Configure logging for this module
log_file = "professional_summary_refinement.log"
try:
    with open(log_file, "w"):
        pass
except OSError as e:
    print(f"Warning: Could not clear log file {log_file}: {e}")

# Ensure handlers are not duplicated if script is run multiple times in same process
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # ch = logging.StreamHandler()  # Optional console handler
    # ch.setFormatter(formatter)
    # logger.addHandler(ch)


# --- Pydantic Models for Evaluation Tool ---

class LengthCheck(BaseModel):
    pass_field: bool = Field(
        ..., alias="pass", description="Does the summary have fewer than 5 lines?"
    )
    comment: str = Field(
        ..., description="Feedback if the summary exceeds the line limit."
    )


class ContentAlignment(BaseModel):
    pass_field: bool = Field(
        ..., alias="pass",
        description="Does the summary cover the key content from the full job description?"
    )
    comment: str = Field(
        ..., description="Feedback on missing or insufficiently addressed content."
    )


class BestPracticesCheck(BaseModel):
    pass_field: bool = Field(
        ..., alias="pass",
        description="Does the summary follow expert writing best practices for tone, clarity, "
                    "specificity, and business relevance?"
    )
    comment: str = Field(
        ..., description="Feedback on which writing principles are not followed properly."
    )


class ProfessionalSummaryEvaluation(BaseModel):
    length_check: LengthCheck
    content_alignment: ContentAlignment
    best_practices_check: BestPracticesCheck


# --- Main Class ---

class ProfessionalSummaryTailor:
    """
    Generates, evaluates, and refines the professional summary section of a resume.
    """
    MAX_REFINEMENT_ITERATIONS = 5
    log_file = log_file  # Make log file path accessible

    def __init__(self, llm_provider: str, api_key: Optional[str] = None):
        """
        Initializes the tailor.

        Args:
            llm_provider: The LLM provider ('openai', 'google', 'anthropic', 'claude').
            api_key: The API key for the LLM provider.
        """
        self.model_name, self.llm, self.raw_llm_client = get_llm(
            llm_provider, api_key, return_raw=True
        )
        logger.info(
            f"Initialized ProfessionalSummaryTailor with model: {self.model_name}"
        )
        # Check if the raw client is suitable for tool calling (basic check)
        self.supports_tool_calling = (
            self.raw_llm_client and hasattr(self.raw_llm_client, 'messages') and
            hasattr(self.raw_llm_client.messages, 'create')
        )
        if not self.supports_tool_calling:
            logger.warning(
                f"Raw LLM client for '{llm_provider}' does not appear to support "
                f"the required 'messages.create' method for tool calling. Evaluation step will be skipped."
            )

    def _get_evaluation_tool_schema(self) -> List[Dict[str, Any]]:
        """Returns the Claude-compatible tool schema for summary evaluation."""
        schema = {
            "name": "evaluate_professional_summary",
            "description": "Evaluates whether a professional summary meets key criteria "
                           "for quality and relevance to the job description.",
            "input_schema": ProfessionalSummaryEvaluation.model_json_schema()
        }
        # Relying on Pydantic's schema generation including alias handling.
        return [schema]

    def _run_initial_generation(
            self, analysis: Dict[str, Any], original_summary: str
    ) -> str:
        """Generates the initial professional summary."""
        logger.info("Starting initial professional summary generation.")
        # TODO: Add 'missing' keywords calculation if needed by the prompt
        inputs = {
            "business": analysis.get("business", "N/A"),
            "responsibilities": analysis.get("responsibilities", []),
            "keywords": analysis.get("keywords", []),
            "tech_stack": analysis.get("tech_stack", []),
            "experience_focus": analysis.get("experience_focus", []),
            "domain_knowledge": analysis.get("domain_knowledge", []),
            "soft_skills": analysis.get("soft_skills", []),
            "original_summary": original_summary,
            "missing": []  # Placeholder for missing keywords logic
        }
        input_vars = list(inputs.keys())

        # Use run_llm_chain for the initial generation
        initial_summary = run_llm_chain(
            self.llm, PROFESSIONAL_SUMMARY_PROMPT, input_vars, inputs, fail_softly=True
        )
        logger.info(f"Initial summary generated:\n{initial_summary}")
        return initial_summary

    def _run_evaluation(
            self, summary: str, job_description: str
    ) -> Optional[ProfessionalSummaryEvaluation]:
        """Runs the LLM evaluation using the tool schema."""
        logger.info(f"Starting evaluation for summary:\n{summary}")
        if not self.supports_tool_calling:
            logger.warning("Skipping evaluation as tool calling is not supported by the client.")
            # Return a default "passing" evaluation to prevent unnecessary rewrites
            return ProfessionalSummaryEvaluation(
                length_check=LengthCheck(pass_field=True, comment="Skipped - Tool calling not supported"),
                content_alignment=ContentAlignment(pass_field=True, comment="Skipped - Tool calling not supported"),
                best_practices_check=BestPracticesCheck(pass_field=True, comment="Skipped - Tool calling not supported")
            )

        tools = self._get_evaluation_tool_schema()
        messages = [{
            "role": "user",
            "content": (
                "Please evaluate the following professional summary based on the "
                f"provided job description.\n\nProfessional Summary:\n{summary}\n\n"
                f"Job Description:\n{job_description}"
            )
        }]

        try:
            logger.info(
                f"Evaluation LLM call initiated with {len(messages)} messages and {len(tools)} tools."
            )
            response = self.raw_llm_client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                messages=messages,
                tools=tools,
                tool_choice={"type": "tool", "name": "evaluate_professional_summary"},
            )
            logger.debug(f"Raw evaluation response object: {response}")

            # Extract tool call input
            tool_input = None
            if response.content and hasattr(response.content[0], 'type') and \
               response.content[0].type == 'tool_use':
                tool_input = response.content[0].input
                logger.info(f"Raw evaluation tool input received: {tool_input}")
            else:
                logger.error(f"No valid tool use found in response: {response.content}")
                return None  # Indicate evaluation failure

            # Parse the JSON input from the tool call
            evaluation = ProfessionalSummaryEvaluation(**tool_input)
            logger.info(
                f"Parsed evaluation result: {evaluation.model_dump_json(indent=2)}"
            )
            return evaluation
        except ValidationError as e:
            logger.error(f"Pydantic validation error during evaluation parsing: {e}", exc_info=True)
            logger.error(f"Invalid tool input data received: {tool_input}")
            return None  # Indicate evaluation failure
        except Exception as e:
            logger.error(
                f"Error during summary evaluation LLM call or parsing: {e}", exc_info=True
            )
            return None  # Indicate evaluation failure

    def _run_rewrite(self, summary: str, feedback: str) -> str:
        """Rewrites the summary based on feedback."""
        logger.info(f"Starting rewrite based on feedback:\n{feedback}")
        # Use the imported REWRITE_SUMMARY_PROMPT
        inputs = {"original_summary": summary, "feedback": feedback}
        input_vars = ["original_summary", "feedback"]

        rewritten_summary = run_llm_chain(
            self.llm, REWRITE_SUMMARY_PROMPT, input_vars, inputs, fail_softly=True
        )
        logger.info(f"Rewritten summary generated:\n{rewritten_summary}")
        return rewritten_summary

    def refine_summary(
            self, analysis: Dict[str, Any], original_summary: str, job_description: str
    ) -> str:
        """
        Performs the iterative refinement process for the professional summary.

        Args:
            analysis: The job analysis dictionary.
            original_summary: The original professional summary text.
            job_description: The full job description text.

        Returns:
            The refined professional summary string.
        """
        current_summary = self._run_initial_generation(analysis, original_summary)
        if "[LLM FAILED]" in current_summary:
            logger.error("Initial summary generation failed. Aborting refinement.")
            return original_summary  # Return original if generation fails

        for i in range(self.MAX_REFINEMENT_ITERATIONS):
            logger.info(f"--- Refinement Iteration {i + 1} ---")
            evaluation = self._run_evaluation(current_summary, job_description)

            if not evaluation:
                logger.warning(
                    "Evaluation failed in iteration %d. Returning current summary.", i + 1
                )
                return current_summary

            # Check if all criteria pass
            all_passed = (
                evaluation.length_check.pass_field and
                evaluation.content_alignment.pass_field and
                evaluation.best_practices_check.pass_field
            )

            if all_passed:
                logger.info(
                    "All evaluation criteria passed in iteration %d. Final summary achieved.", i + 1
                )
                return current_summary

            # Aggregate feedback if not all passed
            feedback_comments = []
            if not evaluation.length_check.pass_field:
                feedback_comments.append(
                    f"- Length: {evaluation.length_check.comment}"
                )
            if not evaluation.content_alignment.pass_field:
                feedback_comments.append(
                    f"- Content Alignment: {evaluation.content_alignment.comment}"
                )
            if not evaluation.best_practices_check.pass_field:
                feedback_comments.append(
                    f"- Best Practices: {evaluation.best_practices_check.comment}"
                )

            aggregated_feedback = "\n".join(feedback_comments)
            if not aggregated_feedback:
                logger.warning(
                    "Evaluation criteria failed but no feedback comments generated in iteration %d. "
                    "Stopping loop.", i + 1
                )
                return current_summary

            logger.info(
                f"Iteration {i + 1} feedback for rewrite:\n{aggregated_feedback}"
            )

            # Rewrite summary based on feedback
            rewritten_summary = self._run_rewrite(
                current_summary, aggregated_feedback
            )
            if "[LLM FAILED]" in rewritten_summary:
                logger.error("Rewrite failed in iteration %d. Aborting refinement.", i + 1)
                return current_summary  # Return last good summary if rewrite fails
            current_summary = rewritten_summary

        logger.warning(
            f"Max refinement iterations ({self.MAX_REFINEMENT_ITERATIONS}) reached. "
            f"Returning last generated summary."
        )
        return current_summary
