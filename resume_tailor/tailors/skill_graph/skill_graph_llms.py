import logging

logger = logging.getLogger(__name__)

def standardize_skills(value_dict, raw_client):
    tool_name = "standardize_header_name"

    prompt = (
        "You are a domain-aware naming standardizer for technical taxonomies.\n\n"
        "You will receive a dictionary where each key is a raw header proposal, and the value is a list of original tags matched to it (the array might be empty).\n\n"
        "Your task is:\n"
        "- Review if the header is already canonical, short, well-formatted, and well-known.\n"
        "- If yes, keep it as-is.\n"
        "- If no, suggest a standardized name that aligns with professional usage with its well known name (e.g. 'google bigquery', 'ci/cd').\n"
        "- If the header is not a skill or it is a soft skill, return empty string.\n"
        "Note that, you have to suggest a well defined standardized name.\n\n"
        "Return a dictionary where **each key is the raw header**, and **each value is a single best canonical string**.\n"
        "Do not modify the keys — only fix the values.\n"
        f"DICTIONARY OF VALUES: {value_dict}"
    )

    tool_schema = [{
        "name": tool_name,
        "description": (
            "Standardize raw headers based on their matched original tag variants based on stackoverflow.\n"
            "Return a dictionary where each key is the raw input header (as given), "
            "and the value is a single canonical string representing the standardized skill name.\n"
            "If the raw name is already canonical or it is a soft skill, return it as-is. "
        ),
        "input_schema": {
            "type": "object",
            "patternProperties": {
                "^[a-zA-Z0-9 _\\-]{3,50}$": {
                    "type": "string",
                    "description": "Your proposed tag to standardize the skill name.",
                }
            },
            "additionalProperties": False
        }
    }]

    try:
        logger.info(f"Sending {len(value_dict)} headers for standardization using model 'claude-3-7-sonnet-20250219'...")

        response = raw_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            messages=[{"role": "user", "content": prompt}],
            tools=tool_schema,
            max_tokens=2048,
            thinking={
                "type": "enabled",
                "budget_tokens": 1024
            },
            temperature=1,
            # tool_choice={"type": "tool", "name": tool_name}
        )

        # Directly access content[2].input as you confirmed
        result_dict = response.content[2].input

        if not isinstance(result_dict, dict):
            logger.error(f"Expected dictionary at response.content[2].input, got {type(result_dict)} instead.")
            return {k: k for k in value_dict.keys()}

        logger.info(f"Successfully standardized {len(result_dict)} headers.")
        return result_dict

    except IndexError:
        logger.error("response.content[2] is missing. Claude tool response might be malformed or incomplete.")
    except AttributeError as e:
        logger.error(f"Missing expected .input field in response.content[2]: {e}")
    except Exception as e:
        logger.exception("Unexpected error during standardization")

    # Fallback if anything goes wrong
    return {k: k for k in value_dict.keys()}
