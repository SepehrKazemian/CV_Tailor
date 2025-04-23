import numpy as np

def find_closest_tags_geo(
    model,
    query: str,
    priors: list[str],
    tag_vecs,
    tag_names,
    alpha: float = 0.7,    # weight on sim_query
    beta:  float = 0.3,    # weight on sim_prior
    top_k: int = 10,
):
    # — 1) encode query & priors —
    q_vec = model.encode(f"query: {query}", normalize_embeddings=True)
    p_vecs = [
        model.encode(f"query: {p}", normalize_embeddings=True)
        for p in priors
    ]
    prior_cent = np.mean(p_vecs, axis=0)
    prior_cent /= np.linalg.norm(prior_cent)

    # — 2) compute similarities —
    sim_q     = tag_vecs @ q_vec         # [n_tags]
    sim_prior = tag_vecs @ prior_cent    # [n_tags]

    # avoid zeros for log
    eps = 1e-6
    sim_q     = np.clip(sim_q,     eps, None)
    sim_prior = np.clip(sim_prior, eps, None)

    # — 3) geometric fusion —
    log_comb  = alpha * np.log(sim_q) + beta * np.log(sim_prior)
    combined  = np.exp(log_comb)

    # — 4) prepare for on‐the‐fly filtering —
    n_query_words = len(query.split())

    # sort all tags by combined score descending
    sorted_idxs = np.argsort(-combined)

    # walk down the sorted list, keep only those ≤ word‐count, until top_k
    results = []
    for idx in sorted_idxs:
        tag = tag_names[idx]
        if len(tag.replace("-", " ").split()) <= n_query_words:
            results.append({
                "tag":            tag,
                "combined_score": float(combined[idx]),
                "sim_query":      float(sim_q[idx]),
                "sim_prior":      float(sim_prior[idx]),
            })
            if len(results) >= top_k:
                break

    return results
