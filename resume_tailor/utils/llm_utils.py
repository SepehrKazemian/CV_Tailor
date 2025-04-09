import logging
import time
from typing import Dict, List, Any

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)


def run_llm_chain(
    llm,
    template_str: str,
    input_vars: List[str],
    inputs: Dict[str, Any],
    retries: int = 2,
    delay: float = 1.5,
    fail_softly: bool = False,
) -> str:
    """
    Run a reusable LLMChain with a given prompt template and inputs.

    Args:
        llm: Language model instance to use.
        template_str (str): The prompt template string.
        input_vars (List[str]): List of input variable names used in the prompt.
        inputs (Dict[str, Any]): Dictionary of input values for the prompt.

    Returns:
        str: The result of the LLMChain output.
    """
    prompt_template = PromptTemplate(template=template_str, input_variables=input_vars)

    # Preprocess inputs (convert lists to joined strings)
    processed_inputs = {
        k: ", ".join(v) if isinstance(v, list) else v for k, v in inputs.items()
    }

    chain = LLMChain(llm=llm, prompt=prompt_template)
    for attempt in range(retries + 1):
        try:
            chain = LLMChain(llm=llm, prompt=prompt_template)
            return chain.run(**processed_inputs).strip()
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries:
                time.sleep(delay)
            elif fail_softly:
                return f"[LLM FAILED]: {str(e)}"
            else:
                raise RuntimeError(f"LLMChain failed after {retries + 1} attempts: {e}")
