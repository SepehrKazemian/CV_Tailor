from typing import Dict, List, Set, Any
from resume_tailor.utils.prompt_templates import PAGE_LIMIT_PROMPT
from resume_tailor.utils.llm_utils import run_llm_chain
import logging

logger = logging.getLogger(__name__)

# Predefined headers for the skills section
SKILL_SECTION_HEADERS = [
    "LLM Ecosystem & GenAI",
    "ML Frameworks & Algorithms",
    "NLP & Text Processing",
    "Computer Vision",
    "Cloud Platforms & Services",
    "MLOps & Model Deployment",
    "Data Engineering & Warehousing",
    "Software Engineering & APIs",
    "Explainability & Evaluation",
    "Document Intelligence",
    "Synthetic Data & Augmentation",
    "Experimentation & Optimization",
    "Monitoring & Observability",
    "Dashboards & Interfaces",
    "Version Control & DevOps",
    "Streaming & Queuing Systems",
    "Security & Compliance",
    "Soft Skills"  # Mandatory section
]

TECH_SUBS = {
    "aws": ["gcp", "azure", "cloud"],
    "gcp": ["aws", "azure", "cloud"],
    "azure": ["aws", "gcp", "cloud"],
    "python": ["java", "c++", "golang", "javascript"],
    "tensorflow": ["pytorch", "keras", "mxnet"],
    "pytorch": ["tensorflow", "keras", "mxnet"],
    "react": ["angular", "vue", "svelte"],
    "javascript": ["typescript", "python", "java"],
    "docker": ["kubernetes", "containerization"],
    "kubernetes": ["docker", "containerization"],
    "sql": ["postgresql", "mysql", "oracle", "nosql"],
    "nosql": ["mongodb", "dynamodb", "cassandra"],
    "fastapi": ["flask", "django", "express"],
    "langchain": ["llamaindex", "semantic-kernel"],
    "llamaindex": ["langchain", "semantic-kernel"],
    "openai": ["anthropic", "gemini", "llama", "mistral"],
    "gemini": ["openai", "anthropic", "llama", "mistral"],
    "anthropic": ["openai", "gemini", "llama", "mistral"],
    "mlflow": ["wandb", "tensorboard", "neptune"],
    "airflow": ["dagster", "prefect", "luigi"],
    "dagster": ["airflow", "prefect", "luigi"],
    "snowflake": ["bigquery", "redshift", "databricks"],
    "bigquery": ["snowflake", "redshift", "databricks"],
    "databricks": ["snowflake", "bigquery", "redshift"],
    "streamlit": ["dash", "gradio", "panel"],
    "tableau": ["power bi", "looker", "quicksight"],
    "power bi": ["tableau", "looker", "quicksight"],
    "elasticsearch": ["opensearch", "solr", "algolia"],
    "qdrant": ["pinecone", "weaviate", "milvus", "vector database"],
    "pinecone": ["qdrant", "weaviate", "milvus", "vector database"],
    "weaviate": ["qdrant", "pinecone", "milvus", "vector database"],
}


def ensure_page_limit(resume: str, llm: Any, warnings_list: List[str]) -> str:
    """
    Ensure the resume fits within 2 pages.

    Args:
        resume: The complete resume text
        llm: The language model instance.
        warnings_list: A list to append warnings to.

    Returns:
        Resume text that fits within 2 pages
    """
    chars_per_page = 3500
    max_chars = chars_per_page * 2

    if len(resume) <= max_chars:
        return resume

    logger.info("Resume exceeds approximate page limit, attempting to condense.")
    input_vars = ["resume", "max_chars"]
    inputs = {"resume": resume, "max_chars": max_chars}
    result = run_llm_chain(
        llm=llm,
        template_str=PAGE_LIMIT_PROMPT,
        input_vars=input_vars,
        inputs=inputs,
        fail_softly=True
    )

    if result.startswith("[LLM FAILED]"):
        warnings_list.append(
            f"Failed to condense resume. Manual review recommended: {result}"
        )
        logger.warning(f"Condensing failed: {result}. Truncating resume.")
        return resume[:max_chars] + "\n\n[Resume truncated to fit 2 pages]"
    elif len(result.strip()) > max_chars:
        warnings_list.append(
            "Resume still exceeds 2 pages after condensing. Manual review recommended."
        )
        logger.warning("Condensing finished, but resume still seems too long.")

    logger.info("Successfully condensed resume.")
    return result.strip()


def flatten_values(d: Dict[str, List[str]]) -> Set[str]:
    """
    Flattens and lowercases all values in a dictionary of lists.
    """
    return set(
        item.strip().lower()
        for v in d.values() if isinstance(v, list)
        for item in v if isinstance(item, str)
    )


def find_missing_keywords(
        cv_keywords: Dict[str, List[str]], job_keywords: Dict[str, List[str]]
) -> Set[str]:
    """
    Finds keywords present in job description but missing from CV.
    """
    cv_set = flatten_values(cv_keywords)
    job_set = flatten_values(job_keywords)
    missing = job_set - cv_set
    logger.info(f"Found {len(missing)} missing keywords between job and CV.")
    return missing
