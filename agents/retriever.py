# app/agents/retriever.py

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_from_disk

# Constants (adjust paths if needed)
INDEX_PATH = "data/vector_index/faiss_hotpot_ip.index"
LOOKUP_PATH = "data/vector_index/doc_lookup.pkl"
DOCS_PATH = "data/hotpot_docs"
EMBED_MODEL_NAME = "BAAI/bge-large-en"

# Load everything once (shared across requests)
_index = faiss.read_index(INDEX_PATH)
with open(LOOKUP_PATH, "rb") as f:
    _doc_lookup = pickle.load(f)
_dataset = load_from_disk(DOCS_PATH)
_embed_model = SentenceTransformer(EMBED_MODEL_NAME)