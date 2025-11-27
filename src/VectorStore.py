import numpy as np
from pathlib import Path
import json
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


def vector_store_config():
    chunks_path = Path("../data/chunks.json")

    with open(chunks_path, "r", encoding="utf-8") as file:
        chunks = json.load(file)
    docs = []
    for chunk in chunks:
        vector = embedding_model.embed_query(chunk['text'])
        docs.append(
            {
                "id": chunk['chunk_id'],
                "vector": vector,
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
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="bioactives_collection",
        embedding=embedding_model,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
    )
    vector_store.add_documents(docs)