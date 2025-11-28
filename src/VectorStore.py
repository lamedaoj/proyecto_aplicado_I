import numpy as np
from pathlib import Path
import json
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer


def vector_store_config():
    chunks_path = Path("../data/chunks.json")

    with open(chunks_path, "r", encoding="utf-8") as file:
        chunks = json.load(file)
    docs = []

    embedding_model = SentenceTransformer(
    "mistralai/Mistral-Small-Embeddings-v0.1"
    )
    for chunk in chunks:
        docs.append(
            {
                "id": chunk['chunk_id'],
                "payload": {
                    "text": chunk['text'],
                    "metadata": {'order':chunk['order'], 'doc_id':chunk['doc_id']}
                }
            }
        )

    client = QdrantClient(path="../data/tmp/langchain_qdrant")

    for col in client.get_collections().collections:
        client.delete_collection(collection_name=col.name)

    client.create_collection(
    collection_name="bioactives_collection",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name="bioactives_collection",
        embedding=embedding_model
    )
    vector_store.add_documents(docs)