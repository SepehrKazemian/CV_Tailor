import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import resume_tailor.tailors.skills.tags.tag_config as tcfg

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def compute_tag_embeddings(tag_counts, model_name="intfloat/e5-base-v2", normalize=True):
    """
    Computes or loads tag embeddings and their log-count-weighted centroid.
    Returns a dict with keys: tag_names, tag_vecs, log_weights, weighted_centroid
    """
    bundle_path = tcfg.dataset_folder / f"{model_name.split('/')[-1]}_tag_embedding_bundle.pkl"

    # ─── Load If Exists ──────────────────────────────────────────────────────
    if bundle_path.exists():
        print("✅ Loaded tag embedding bundle from cache.")
        bundle = load_pickle(bundle_path)
        return bundle.values()

    # ─── Compute ────────────────────────────────────────────────────────────
    tag_names = list(tag_counts.keys())
    log_weights = np.log1p([tag_counts[t] for t in tag_names])
    log_weights = log_weights / np.sum(log_weights)

    print(f"🚀 Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print("🔍 Embedding all tags...")
    tag_vecs = model.encode(tag_names, normalize_embeddings=normalize)

    weighted_centroid = (tag_vecs.T @ log_weights).T

    # ─── Save ───────────────────────────────────────────────────────────────
    bundle = {
        "tag_names": tag_names,
        "tag_vecs": tag_vecs,
        "log_weights": log_weights,
        "weighted_centroid": weighted_centroid
    }
    save_pickle(bundle, bundle_path)
    print("💾 Saved tag embedding bundle to cache.")

    return list(bundle.values())
