import os
import sys
import argparse
import logging
import importlib.util
from pathlib import Path
from resume_tailor import ResumeTailor

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config():
    """
    Dynamically load the config.py module if it exists.

    Returns:
        module: The loaded config module or a default fallback object.
    """
    config_path = Path(__file__).resolve().parent / "config.py"
    try:
        spec = importlib.util.spec_from_file_location("config", config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        return config
    except Exception:
        config = type("", (), {})()
        config.LLM = "anthropic"
        config.version = None
        config.key = None
        return config


def get_api_key(llm: str, env_var_map: dict, config) -> bool:
    """
    Ensure the appropriate API key is set in the environment.

    Args:
        llm (str): The LLM provider.
        env_var_map (dict): Mapping of LLM names to environment variables.
        config (module): Configuration module.

    Returns:
        bool: True if the API key is available, False otherwise.
    """
    key_env = env_var_map.get(llm)
    if key_env in os.environ:
        return True

    if llm in ["anthropic", "claude"]:
        api_key = getattr(config, "key", None)
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
            return True
    return False


def read_job_description(job_file: str, job_text: str) -> str:
    """
    Retrieve job description from a file or direct text input.

    Args:
        job_file (str): Path to the job description file.
        job_text (str): Direct job description input.

    Returns:
        str: Job description content.
    """
    if job_file:
        try:
            with open(job_file, "r") as file:
                return file.read()
        except FileNotFoundError:
            logger.error(f"Job description file '{job_file}' not found.")
            sys.exit(1)
    return job_text


def ensure_output_path(output_dir: str, filename: str, fmt: str) -> str:
    """
    Construct the full path for the output file.

    Args:
        output_dir (str): Output directory.
        filename (str): Output file name.
        fmt (str): Output file format.

    Returns:
        str: Full output path with proper extension.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not filename.endswith(f".{fmt}"):
        filename = f"{filename.rsplit('.', 1)[0]}.{fmt}"
    return os.path.join(output_dir, filename)


def parsing_args():
    parser = argparse.ArgumentParser(
        description="Tailor your resume based on a job description."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--job-file", type=str, help="Path to a file containing the job description"
    )
    input_group.add_argument("--job-text", type=str, help="Job description text")

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory (default: output)",
    )
    parser.add_argument(
        "--output", type=str, default="tailored_resume.docx", help="Output file name"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["docx", "txt", "md"],
        default="docx",
        help="Output format",
    )
    parser.add_argument(
        "--llm",
        type=str,
        choices=["openai", "google", "anthropic", "claude"],
        default=None,
        help="LLM provider to use (default: from config.py)",
    )
    parser.add_argument(
        "--api-key", type=str, help="API key for the selected LLM provider"
    )
    return parser


def main():
    """
    Main entry point for the resume tailoring CLI.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    parser = parsing_args()
    args = parser.parse_args()

    env_var_map = {
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }

    config = load_config()
    args.llm = (args.llm or getattr(config, "LLM", "anthropic")).lower()
    llm_key_check = "anthropic" if args.llm == "claude" else args.llm

    if args.api_key:
        os.environ[env_var_map.get(llm_key_check, "")] = args.api_key

    if not get_api_key(llm_key_check, env_var_map, config):
        logger.error(
            f"{args.llm.capitalize()} API key not found. Use --api-key or set it in config.py or env."
        )
        return 1

    job_description = read_job_description(args.job_file, args.job_text)
    output_path = ensure_output_path(args.output_dir, args.output, args.format)

    try:
        logger.info(f"Analyzing job description using {args.llm.capitalize()}...")
        tailor = ResumeTailor(cv_dir="cv", llm_provider=args.llm, api_key=args.api_key)
        tailored_resume = tailor.generate_tailored_resume(job_description)
        tailor.save_resume(tailored_resume, output_path, args.format)
        logger.info(f"Success! Tailored resume saved to: {output_path}")
        return 0
    except Exception as e:
        logger.exception("Error tailoring resume", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())