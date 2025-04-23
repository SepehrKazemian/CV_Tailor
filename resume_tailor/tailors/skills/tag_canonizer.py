
from resume_tailor.tailors.skills.tags.load_tags_stackoverflow import parse_stackoverflow_tags
from resume_tailor.tailors.skills.tags.tag_matching import generate_subphrases, match_tags, build_acronym_map
from resume_tailor.tailors.skills.tags.tag_utils import apply_stems_or_lemmas, _normalize
from resume_tailor.tailors.skills.tags.tag_embedding_loader import compute_tag_embeddings
from sentence_transformers import SentenceTransformer
from resume_tailor.tailors.skills.tags.tag_embedding_finder import find_closest_tags_geo
import re
from typing import List, Dict

def filter_candidates_by_header(header: str, candidates: list[str]) -> list[str]:
    """
    Keep only those candidate tags whose every sub-token
    (split on non-alphanumerics) appears in the header, after lemmatization.
    """
    # 1) build a set of lemmatized header-words
    hdr_words = set(
        _normalize(t, "lemmatize")
        for t in re.findall(r"\w+", header)
    )
    out = []
    for cand in candidates:
        toks = [t for t in re.split(r"\W+", cand) if t]
        # lemmatize each token, then check membership
        if all(_normalize(tok) in hdr_words for tok in toks):
            out.append(cand)
    return out
    

def tag_matching_stackoverflow(skill_headers, tag_set):
    dict_hdr_trim = {}
    for hdr in skill_headers:
        hdr = hdr.replace("/", "")
        subs = generate_subphrases(hdr)
        lemmas = apply_stems_or_lemmas(subs)
        main_cands = subs + lemmas
        matched_main = set(match_tags(main_cands, tag_set))

        acronym_map = build_acronym_map(subs)
        matched_acrs = set(match_tags(list(acronym_map.keys()), tag_set))
        # 6) Filter out any acronym whose origin was already matched
        filtered_acrs = {
            acr for acr in matched_acrs
            if acronym_map[acr].lower().replace(" ", "-") not in matched_main
        }

        results = list(matched_main | filtered_acrs)
        
        dict_hdr_trim[hdr] = [i.replace("-", " ") for i in results]
    
    return dict_hdr_trim


def tag_embedding_search(skill_headers_map, model, tag_names, tag_vecs):
    dict_hdr_trim_embedding = {}
    for h, val_list in skill_headers_map.items():
        if len(val_list) == 0:   continue
        
        h = h.replace("/", "")
        variation_list = apply_stems_or_lemmas([h], _stemming=False)
        variation_list = set(variation_list + [h])
        dict_hdr_trim_embedding[h] = []
        for v in variation_list:
            print(v)
            results = find_closest_tags_geo(
                model = model,
                query = v,
                priors=val_list,
                tag_vecs = tag_vecs,
                tag_names = tag_names,
                alpha=0.7,   # 70% emphasis on matching your actual query
                beta=0.3,    # 30% emphasis on matching your priors (hard requirement)
                top_k=10
            )
            print(h, results)
            dict_hdr_trim_embedding[h].extend([i["tag"].replace("-", " ") for i in results])
            
    return dict_hdr_trim_embedding
            
def pick_best_candidates(header: str, candidates: List[str]) -> List[str]:
    """
    From a list of candidates, return all those whose tokens
    achieve the longest run of *consecutive* (substring) matches
    in the header.
    """
    # tokenize the header just once
    hdr_tokens = re.findall(r"\w+", header.lower())
    best_score = 0
    best_cands: List[str] = []

    for cand in candidates:
        # split candidate into tokens
        cand_tokens = re.findall(r"\w+", cand.lower())
        L = len(cand_tokens)
        score = 0

        # slide a window of size L over the header tokens
        for i in range(len(hdr_tokens) - L + 1):
            ok = True
            for j in range(L):
                ht = hdr_tokens[i + j]
                ct = cand_tokens[j]
                # allow substring match in either direction
                if ct not in ht and ht not in ct:
                    ok = False
                    break
            if ok:
                score = L
                break  # full‐length match found

        # update the “best” list
        if score > best_score:
            best_score = score
            best_cands = [cand]
        elif score == best_score and score > 0:
            best_cands.append(cand)

    return best_cands


def tag_matching_main(skill_headers_map, model_name='intfloat/e5-base-v2'):
    tag_set, tag_counts = parse_stackoverflow_tags()
    header_matched_dict = tag_matching_stackoverflow(list(skill_headers_map.keys()), tag_set)
    
    model = SentenceTransformer(model_name)
    tag_names, tag_vecs, log_weights, weighted_centroid = compute_tag_embeddings(tag_counts)
    dict_hdr_trim_embedding = tag_embedding_search(skill_headers_map, model, tag_names, tag_vecs)
    
    # getting rid of trimmed words not in the header
    dict_filtered = {}
    for hdr, cands in dict_hdr_trim_embedding.items():
        unique = list(dict.fromkeys(cands))
        dict_filtered[hdr] = filter_candidates_by_header(hdr, unique)
    
    # matched candidates in both trimming or combined in case of no match 
    dict_merged = {}
    for h, v in dict_filtered.items():
        dict_merged[h] = list(set(v) & set(header_matched_dict[h]))
        if not dict_merged[h]:
            dict_merged[h] = list(set(v) | set(header_matched_dict[h]))
    
    final_mapping: Dict[str, List[str]] = {
        hdr: pick_best_candidates(hdr, cands)
        for hdr, cands in dict_merged.items()
    }
    
    return final_mapping