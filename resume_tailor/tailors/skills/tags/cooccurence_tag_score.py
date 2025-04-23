import numpy as np

def find_closest_tags_faiss_geo_with_graph(
    model,
    query: str,
    priors: list[str],
    tag_norms: list[str],
    tag_embs: np.ndarray,    # [n_tags, dim], L2‑normalized SBERT vectors
    index,                   # FAISS IndexFlatIP built on tag_embs
    co_graph: dict,          # nested dict: co_graph[tag1][tag2] = count
    alpha: float = 0.6,      # weight on SBERT sim(query, tag)
    beta:  float = 0.2,      # weight on SBERT sim(prior_cent, tag)
    gamma: float = 0.2,      # weight on co‑occurrence count signal
    top_k: int = 5,
    shortlist_k: int = 200
):
    # 1) normalize & embed query + priors
    q = query.lower().replace(" ", "-")
    qv = model.encode([q], normalize_embeddings=True)[0]
    
    if not priors:
        prior_cent = np.zeros(tag_embs.shape[1], dtype=float)
    else:
        pvs = [model.encode([p.lower().replace(" ", "-")], normalize_embeddings=True)[0]
            for p in priors]
        prior_cent = np.mean(pvs, axis=0)
        prior_cent /= np.linalg.norm(prior_cent)

    # 2) get shortlist by SBERT query similarity
    Dq, Iq = index.search(qv.reshape(1, -1), shortlist_k)
    sims_q     = Dq[0]                              # [shortlist_k]
    sims_prior = tag_embs[Iq[0]] @ prior_cent       # [shortlist_k]

    # 3) build a raw co‑occurrence score for each candidate
    #    e.g. sum of counts between each prior and the candidate
    sims_graph = []
    for idx in Iq[0]:
        tag = tag_norms[idx]
        s = 0
        for p in priors:
            p_norm = p.lower().replace(" ", "-")
            s += co_graph.get(p_norm, {}).get(tag, 0)
        sims_graph.append(s)
    sims_graph = np.array(sims_graph, dtype=float)

    # normalize each signal to [0,1]
    def normalize(arr):
        if arr.max() == 0:
            return arr
        return arr / arr.max()
    nq = normalize(sims_q)
    np_ = normalize(sims_prior)
    ng = normalize(sims_graph)

    # 4) fuse them linearly (you can also try log‐fusion here)
    fused = alpha * nq + beta * np_ + gamma * ng

    # 5) pick top_k after optional length filter
    n_qwords = len(query.split())
    results = []
    for score, idx in sorted(zip(fused, Iq[0]), reverse=True):
        tag = tag_norms[idx]
        # enforce no multi‑word tags longer than query
        if len(tag.replace("-", " ").split()) <= n_qwords:
            results.append({
                "tag":            tag,
                "fused_score":    float(score),
                "sim_query":      float(sims_q[Iq[0]==idx][0]),
                "sim_prior":      float(sims_prior[Iq[0]==idx][0]),
                "sim_graph":      float(sims_graph[Iq[0]==idx][0])
            })
            if len(results) >= top_k:
                break

    return results

def resolve_tag(phrase, model, tag_norms, index, k=10, threshold=0.6):
    # 1. normalize the input the same way
    q = phrase.lower().replace(" ", "-").replace("/", "-")
    # 2. embed & normalize
    qv = model.encode([q], normalize_embeddings=True)
    # 3. search FAISS for the top-k closest
    D, I = index.search(qv, k)   # D = [ [score1, score2, …] ], I = [ [idx1, idx2, …] ]

    # 4. build a list of (tag, score) for those above threshold
    results = []
    for idx, score in zip(I[0], D[0]):
        if score >= threshold:
            results.append((tag_norms[idx], float(score)))

    return results