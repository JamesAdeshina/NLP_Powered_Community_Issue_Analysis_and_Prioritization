from models.load_models import get_embeddings_model
import torch
from sentence_transformers import util


def get_embeddings(texts):
    model = get_embeddings_model()
    return model.encode(texts, convert_to_tensor=True)


def semantic_search(query, documents, top_k=3):
    model = get_embeddings_model()
    query_embedding = model.encode(query, convert_to_tensor=True)
    doc_embeddings = model.encode(documents, convert_to_tensor=True)

    cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(top_k, len(documents)))

    return top_results