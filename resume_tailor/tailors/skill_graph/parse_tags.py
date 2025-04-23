from resume_tailor.tailors.skill_graph.tag_matcher import StackOverflowTagger
from resume_tailor.tailors.skills.tags.tag_embedding_loader import compute_tag_embeddings
from resume_tailor.tailors.skills.tags.load_tags_stackoverflow import parse_stackoverflow_tags

# sot = StackOverflowTagger(normalized_skills)
tag_set, tag_counts = parse_stackoverflow_tags()
tag_names, tag_vecs, log_weights, weighted_centroid = compute_tag_embeddings(tag_counts)

import numpy as np
from sentence_transformers import SentenceTransformer

tag_names = normalized_skills
model = SentenceTransformer("intfloat/e5-base-v2")

print("🔍 Embedding all tags...")
skills_dict = {i.replace("/", "-").replace(" ", "-"):i for i in normalized_skills}
skill_vecs = model.encode(list(skills_dict.keys()), normalize_embeddings=True)

sot = StackOverflowTagger()
matches = sot.get_matches(list(skills_dict.keys()), skill_vecs)

from fuzzywuzzy import fuzz
import re

def normalize(text):
    return re.sub(r'[^a-z0-9]', '', text.lower())

def add_fuzzy_matching_to_matches(matches):
    for key, values in matches.items():
        if isinstance(values, list):
            norm_key = normalize(key)
            fuzzy_scores = [fuzz.ratio(norm_key, normalize(value)) for value in values]
            matches[key] = {
                "original": values,
                "fuzzy_scores": fuzzy_scores
            }
    return matches
# Example usage
matches_ratio = add_fuzzy_matching_to_matches(matches)
matches_ratio


matches_with_100 = []
matches_with_values = []
matches_without_values = []

for key, value in matches_ratio.items():
    if "fuzzy_scores" in value and 100 in value["fuzzy_scores"]:
        index = value["fuzzy_scores"].index(100)
        matches_with_100.append({key: value["original"][index]})
    elif "original" in value and value["original"]:
        matches_with_values.append({key: value["original"]})
    else:
        matches_without_values.append({key: []})

# Output the three lists
standard_tags_1 = standardize_skills_L1(matches_with_values)
standard_tags_2 = standardize_skills_L1(matches_without_values)

standard_tags = standard_tags_1 + standard_tags_2 + matches_with_100
# TODO:
# builder._normalize_skill_list(standard_tags.values())
# add the new values to tag_names, tag_vecs
# save the original tag value and converted so when we see similar skills we could map it to these tags

def standardize_skills_L1(value_dict):
    tool_name = "standardize_header_name"

    prompt = f"""
    You are a domain-aware naming standardizer for technical taxonomies.

    You will receive a dictionary where each key is a raw header proposal, and the value is a list of original tags matched to it (the array might be empty).

    Your task is:
    - Review if the header is already canonical, short, well-formatted, and well-known.
    - If yes, keep it as-is.
    - If no, suggest a standardized name that aligns with professional usage with it's well known name (e.g. 'google bigquery', 'ci/cd').
    - If the header is not a skill or it is a soft skill, return empty string.
    Note that, you have to suggest a well defined standardized name.

    Return a dictionary where **each key is the raw header**, and **each value is a single best canonical string**.
    Do not modify the keys — only fix the values.
    DICTIONARY OF VALUES: {value_dict}
    """

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
    
    return response.content[2].input

