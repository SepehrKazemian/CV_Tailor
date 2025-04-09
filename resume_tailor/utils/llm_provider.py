"""
LLM Provider Utility Module

This module handles the initialization of different LLM providers.
"""

import os
from typing import Optional, Any, Tuple
import openai
import anthropic
# LLM providers
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

# Import config.py dynamically
import importlib.util
from pathlib import Path

# Try to import config.py
try:
    config_path = Path(__file__).resolve().parents[2] / "config.py"
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
except Exception:
    # If config.py can't be imported, use default values
    config = type('', (), {})()
    config.LLM = "anthropic"
    config.version = None
    config.key = None


def get_llm(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    return_raw: bool = False  # Keep arg for compatibility, but behavior changed
) -> Tuple[str, Any, Optional[Any]]:
    """
    Initialize an LLM based on the provider and return model name, LangChain wrapper, and raw client.

    Args:
        provider: LLM provider ('openai', 'google', 'anthropic', 'claude', or None to use config).
        api_key: API key for the selected LLM provider (uses config or env var if None).
        return_raw: If True, ensures raw client is attempted to be initialized (always returned now).

    Returns:
        Tuple[str, Any, Optional[Any]]: (model_name, langchain_llm_wrapper, raw_client_or_none)

    Raises:
        ValueError: If an unsupported LLM provider is specified.
    """
    provider = (provider or getattr(config, 'LLM', 'anthropic')).lower()
    api_key = api_key or getattr(config, 'key', None)
    model_version = getattr(config, 'version', None)

    model_name = ""
    langchain_llm = None
    raw_client = None

    final_api_key = None  # Determine the final API key to use

    if provider == "openai":
        final_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        model_name = "gpt-4" if not model_version else f"gpt-{model_version}"
        langchain_llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,
            api_key=final_api_key
        )
        # Initialize raw client if API key is available
        if final_api_key:
            try:
                raw_client = openai.OpenAI(api_key=final_api_key)
            except Exception as e:
                print(f"Warning: Could not initialize raw OpenAI client: {e}")
        else:
            print("Warning: OpenAI API key not found, cannot initialize raw client.")

    elif provider == "google":
        final_api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        model_name = "gemini-pro" if not model_version else f"gemini-{model_version}"
        langchain_llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.2,
            api_key=final_api_key
        )
        # Raw Google client initialization might require google.generativeai package
        raw_client = None
        print("Warning: Raw Google Gemini client initialization not implemented.")

    elif provider in {"anthropic", "claude"}:
        final_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        # Determine model name based on version from config
        if model_version:
            if isinstance(model_version, (int, float)):
                # Map numeric versions if needed, otherwise use default
                if model_version == 3.5:
                    model_name = "claude-3-haiku-20240307"
                elif model_version == 3.0:
                    model_name = "claude-3-sonnet-20240229"  # Example mapping
                else:
                    model_name = "claude-3-opus-20240229"  # Default Opus
            else:
                model_name = str(model_version)  # Use string version directly
        else:
            model_name = "claude-3-opus-20240229"  # Default if no version specified

        langchain_llm = ChatAnthropic(
            model=model_name,
            temperature=0.2,
            api_key=final_api_key
        )
        # Initialize raw client if API key is available
        if final_api_key:
            try:
                raw_client = anthropic.Anthropic(api_key=final_api_key)
            except Exception as e:
                print(f"Warning: Could not initialize raw Anthropic client: {e}")
        else:
            print("Warning: Anthropic API key not found, cannot initialize raw client.")

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    if not langchain_llm:
        raise ValueError(f"Could not initialize LangChain LLM for provider: {provider}")

    # Always return all three values
    return model_name, langchain_llm, raw_client


def get_provider_env_var(provider: str) -> str:
    """
    Get the environment variable name for the specified LLM provider.

    Args:
        provider: LLM provider ('openai', 'google', 'anthropic', or 'claude')

    Returns:
        The environment variable name for the provider's API key

    Raises:
        ValueError: If an unsupported LLM provider is specified
    """
    provider_lower = provider.lower()
    if provider_lower == "openai":
        return "OPENAI_API_KEY"
    elif provider_lower == "google":
        return "GOOGLE_API_KEY"
    elif provider_lower == "anthropic" or provider_lower == "claude":
        return "ANTHROPIC_API_KEY"
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
