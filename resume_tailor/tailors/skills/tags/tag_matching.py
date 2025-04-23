import re
from resume_tailor.tailors.skills.tags.tag_utils import stem_token, lemmatize_token

def generate_subphrases(header: str) -> list[str]:
    """
    Generate all contiguous subphrases by removing one token
    at a time from either the start or the end.
    E.g. "A B C" → ["A B C", "A B", "A", "B C", "C"]
    """
    tokens = header.split()
    n = len(tokens)
    subs = [header]
    # remove from end
    for i in range(1, n):
        subs.append(" ".join(tokens[:n - i]))
    # remove from front
    for i in range(1, n):
        subs.append(" ".join(tokens[i:]))
    # dedupe while preserving order
    seen = set()
    out = []
    for s in subs:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out

def generate_acronyms(phrases: list[str]) -> list[str]:
    """
    For any multi‑word phrase, generate its acronym.
    E.g. "Machine Learning Frameworks" → "MLF"
    """
    acrs = []
    for ph in phrases:
        tokens = ph.split()
        if len(tokens) > 1:
            acr = "".join(tok[0].upper() for tok in tokens)
            acrs.append(acr)
    return list(dict.fromkeys(acrs))

def build_acronym_map(subs: list[str]) -> dict[str, str]:
    """
    Given your list of sub‑phrases, return a map:
      normalized_acronym -> original sub‑phrase
    E.g. "MLF" -> "Machine Learning Frameworks"
    """
    amap = {}
    for ph in subs:
        for acr in generate_acronyms([ph]):
            norm = acr.lower().replace(" ", "-")
            amap[norm] = ph
    return amap

def filter_acronyms_via_origin(matched: list[str], acronym_map: dict[str, str]) -> list[str]:
    """
    Drop any acronym tag if its origin phrase also matched.
    Keep everything else.
    """
    out = []
    for tag in matched:
        # is this tag one of our acronyms?
        if tag in acronym_map:
            # get the origin phrase, normalize
            origin_norm = acronym_map[tag].lower().replace(" ", "-")
            # if we also matched that origin, skip the acronym
            if origin_norm in matched:
                continue
        out.append(tag)
    return out

def match_tags(candidates: list[str], tag_set: set[str]) -> list[str]:
    """
    Canonicalize candidates to SO‑style (lower‑hyphen) and
    return only those that exist in tag_set.
    """
    hits = []
    for cand in candidates:
        normalized = cand.lower().replace(" ", "-")
        if normalized in tag_set:
            hits.append(normalized)
    return hits