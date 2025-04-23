import numpy as np
from langchain.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List


def vector_db(model_name="intfloat/e5-base-v2"):
    # Step 1: Load skills
    skill_doc = []
    with open("/root/Desktop/Sepehr/cv/skills.txt") as f:
        for line in f:
            parts = line.split(":")[1].split(",")
            skill_doc.extend(skill.strip() for skill in parts)

    # Step 2: Convert to LangChain documents
    documents = [Document(page_content=skill) for skill in skill_doc]

    # Step 3: Use normalized HuggingFace embeddings (for cosine similarity)
    class NormalizedHuggingFaceEmbeddings(HuggingFaceEmbeddings):
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            raw = super().embed_documents(texts)
            return [list(np.array(v) / np.linalg.norm(v)) for v in raw]

        def embed_query(self, text: str) -> List[float]:
            vec = super().embed_query(text)
            return list(np.array(vec) / np.linalg.norm(vec))

    embedding = NormalizedHuggingFaceEmbeddings(model_name=model_name)

    # Step 4: Build FAISS vector store with cosine similarity (via inner product)
    vector = FAISS.from_documents(
        documents,
        embedding,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
    )

    # Step 5: Build retriever
    # retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    return vector


# def skill_matcher_tool(
#     skills: List[str],
#     vectorstore: VectorStoreRetriever,
#     threshold: float = 0.8,
#     initial_k: int = 10,
#     max_k: int = 200
# ) -> List[str]:
#     """
#     Matches extracted job skills with the candidate's skills using cosine similarity.
#     Expands top-k search only if best result may still exceed threshold.

#     Args:
#         skills: List of extracted job description skills.
#         vectorstore: FAISS vector store with .similarity_search_with_score.
#         threshold: Cosine similarity threshold for accepting a match.
#         initial_k: Starting value of top-k.
#         max_k: Maximum value of k to consider during expansion.

#     Returns:
#         List of matched candidate skill strings above the threshold.
#     """
#     matched: Set[str] = set()

#     for skill in skills:
#         k = initial_k
#         print(skill)

#         while k <= max_k:
#             results = vectorstore.similarity_search_with_score(skill, k=k)

#             if not results:
#                 break  # No results at all, skip
 
#             # ✅ If top result is already below threshold, we can stop
#             if results[0][1] < threshold:
#                 break

#             found_match = False
            
#             for doc, score in results:
#                 print(doc, score)
#                 if score >= threshold:
#                     matched.add(doc.page_content)
#                     found_match = True

#             if found_match:
#                 break  # ✅ Stop expanding if we got good enough matches

#             k *= 2  # 🔁 Try again with larger k

#     return sorted(matched)