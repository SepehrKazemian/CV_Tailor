import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from resume_tailor.tailors.skills.tags.cooccurence_graph import (
    build_and_save_chunks,
    merge_saved_chunks,
)

# ─── Configuration ────────────────────────────────────────────────────────────

DATASET_DIR       = Path("resume_tailor/dataset")
CO_GRAPH_PATH     = DATASET_DIR / "co_graph.pkl"
MODEL_NAME        = "intfloat/e5-base-v2" # "nomic-ai/nomic-embed-text-v1.5"
BUNDLE_PATH       = DATASET_DIR / f"{MODEL_NAME.split('/')[1]}all_tags_bundle.pkl"
ARCHIVE_PATH      = Path("stackoverflow.com-Posts.7z")
# ─── Helpers for co‑occurrence graph ──────────────────────────────────────────

def load_co_graph(path: Path):
    """Load a previously built co‑occurrence graph, or return None if missing."""
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def build_co_graph(archive_path: Path, out_path: Path):
    """Stream & chunk the archive, merge the results, pickle & return."""
    # 1) build & save intermediate chunks
    build_and_save_chunks(
        archive_path=archive_path,
        lines_per_chunk=20_000_000,
        batch_size=5_000,
        out_folder=DATASET_DIR / "cooc_chunks"
    )
    # 2) merge them into one dict
    co_graph = merge_saved_chunks(DATASET_DIR / "cooc_chunks")
    # 3) persist
    with open(out_path, "wb") as f:
        pickle.dump(dict(co_graph), f)
    return co_graph

def get_co_graph():
    """Load or build the co‑occurrence graph."""
    co_graph = load_co_graph(CO_GRAPH_PATH)
    if co_graph is None:
        co_graph = build_co_graph(ARCHIVE_PATH, CO_GRAPH_PATH)
    return co_graph

# ─── Helpers for tag embedding bundle ──────────────────────────────────────────

def load_bundle(path: Path):
    """Load a prebuilt (tag_norms, tag_embs, index) bundle, or return None."""
    if path.exists():
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        # reconstruct the FAISS index
        index = faiss.deserialize_index(bundle["index"])
        return bundle["tag_norms"], bundle["tag_embs"], index
    return None

def build_bundle(co_graph, path: Path, model):
    """Compute tag_norms, tag_embs, and index; pickle them; return."""
    # normalize & embed
    tag_list  = list(co_graph.keys())
    tag_norms = [t.lower().replace(" ", "-").replace("/", "-") 
                 for t in tag_list]
    
    tag_embs  = model.encode(tag_norms, normalize_embeddings=True)

    # build FAISS
    dim   = tag_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(tag_embs)

    # serialize index
    index_bytes = faiss.serialize_index(index)
    bundle = {
        "tag_norms": tag_norms,
        "tag_embs":  tag_embs,
        "index":     index_bytes
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    return tag_norms, tag_embs, index, model

def get_bundle(co_graph):
    """Load or build the tag embedding + FAISS bundle."""
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    loaded = load_bundle(BUNDLE_PATH)
    if loaded:
        return loaded[0], loaded[1], loaded[2], model
    return build_bundle(co_graph, BUNDLE_PATH, model)

# ─── Public API ───────────────────────────────────────────────────────────────

def initialize_tag_pipeline():
    """
    Ensure co_graph and embedding bundle exist on disk (or build them),
    and return (tag_norms, tag_embs, index, co_graph).
    """
    DATASET_DIR.mkdir(exist_ok=True)
    co_graph   = get_co_graph()
    tag_norms, tag_embs, index, model = get_bundle(co_graph)
    return tag_norms, tag_embs, index, co_graph, model