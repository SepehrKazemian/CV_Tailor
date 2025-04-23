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

# Try to import config.py and credentials.py
try:
    # Config import
    config_path = Path(__file__).resolve().parents[2] / "config.py"
    spec_config = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec_config)
    spec_config.loader.exec_module(config)
except Exception:
    # If config.py can't be imported, use default values
    config = type('', (), {})() # Create an empty object
    config.LLM = "anthropic"
    config.version = None
    # config.key is no longer used here

try:
    # Credentials import
    creds_path = Path(__file__).resolve().parents[2] / "credentials.py"
    spec_creds = importlib.util.spec_from_file_location("credentials", creds_path)
    credentials = importlib.util.module_from_spec(spec_creds)
    spec_creds.loader.exec_module(credentials)
except Exception:
    # If credentials.py can't be imported, create an empty object
    credentials = type('', (), {})() # Create an empty object

# --- Original config fallback logic (modified) ---
try:
    # This block is kept for loading LLM and version, but key is ignored
    config_path = Path(__file__).resolve().parents[2] / "config.py"
    spec = importlib.util.spec_from_file_location("config", config_path)
    # Re-assign config here to ensure it's loaded if credentials failed but config exists
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
except Exception:
    # If config.py still fails, ensure defaults are set on the existing object
    if not hasattr(config, 'LLM'):
        config.LLM = "anthropic"
    if not hasattr(config, 'version'):
        config.version = None
    # No need to set config.key default


def get_llm(
    provider: Optional[str] = None,
    api_key: Optional[str] = None, # Explicitly passed API key takes highest priority
    return_raw: bool = False
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
    # API Key Loading Order: function arg -> credentials.py -> environment variable
    final_api_key = api_key # Start with the explicitly passed key

    # Get the expected key name from credentials.py and env var name
    cred_key_name = None
    env_var_name = None
    if provider == "openai":
        cred_key_name = "OPENAI_API_KEY"
        env_var_name = "OPENAI_API_KEY"
    elif provider == "google":
        cred_key_name = "GOOGLE_API_KEY"
        env_var_name = "GOOGLE_API_KEY"
    elif provider in {"anthropic", "claude"}:
        cred_key_name = "ANTHROPIC_API_KEY"
        env_var_name = "ANTHROPIC_API_KEY"

    # Try loading from credentials.py if not passed explicitly
    if final_api_key is None and cred_key_name:
        final_api_key = getattr(credentials, cred_key_name, None)

    # Try loading from environment variable if still not found
    if final_api_key is None and env_var_name:
        final_api_key = os.environ.get(env_var_name)

    # --- Get model version from config ---
    model_version = getattr(config, 'version', None)

    # --- Initialize LLM ---
    model_name = ""
    langchain_llm = None
    raw_client = None

    if provider == "openai":
        # final_api_key is already determined above
        model_name = "gpt-4" if not model_version else f"gpt-{model_version}"
        langchain_llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,
            api_key=final_api_key # Use the determined key
        )
        # Initialize raw client if API key is available
        if final_api_key:
            try:
                raw_client = openai.OpenAI(api_key=final_api_key)
            except Exception as e:
                print(f"Warning: Could not initialize raw OpenAI client: {e}")
        elif not api_key: # Only warn if key wasn't explicitly passed as None
             print("Warning: OpenAI API key not found in credentials.py or environment variables.")

    elif provider == "google":
        # final_api_key is already determined above
        model_name = "gemini-pro" if not model_version else f"gemini-{model_version}"
        langchain_llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.2,
            api_key=final_api_key # Use the determined key
        )
        # Raw Google client initialization might require google.generativeai package
        raw_client = None # Keep as None for now
        print("Warning: Raw Google Gemini client initialization not implemented.")
        if not final_api_key and not api_key: # Only warn if key wasn't explicitly passed as None
            print("Warning: Google API key not found in credentials.py or environment variables.")

    elif provider in {"anthropic", "claude"}:
        # final_api_key is already determined above
        # Determine model name based on version from config
        if model_version:
            if isinstance(model_version, (int, float)):
                # Map numeric versions if needed, otherwise use default
                if model_version == 3.5:
                    model_name = "claude-3-haiku-20240307"
                elif model_version == 3.0:
                    model_name = "claude-3-sonnet-20240229" # Example mapping
                # Add mapping for 3.7 if needed, otherwise default
                elif model_version == 3.7: # Assuming 3.7 maps to Opus for now
                     model_name = "claude-3-opus-20240229"
                else:
                    model_name = "claude-3-opus-20240229" # Default Opus
            else:
                model_name = str(model_version) # Use string version directly
        else:
            model_name = "claude-3-opus-20240229" # Default if no version specified

        langchain_llm = ChatAnthropic(
            model=model_name,
            temperature=0.2,
            api_key=final_api_key # Use the determined key
        )
        # Initialize raw client if API key is available
        if final_api_key:
            try:
                raw_client = anthropic.Anthropic(api_key=final_api_key)
            except Exception as e:
                print(f"Warning: Could not initialize raw Anthropic client: {e}")
        elif not api_key: # Only warn if key wasn't explicitly passed as None
            print("Warning: Anthropic API key not found in credentials.py or environment variables.")

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
