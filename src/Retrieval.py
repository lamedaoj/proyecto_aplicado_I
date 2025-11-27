from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

def retrieve(
    query: str,
    k: int = 5,
    doc_id: dict | None = None,
):

    client = QdrantClient(path="../data/tmp/langchain_qdrant")

    qdrant = QdrantVectorStore(
        client=client,
        collection_name="bioactives_collection",
        embedding=embedding_model,
    )

    results = qdrant.search(
        query,
        k=k,
        search_type="mmr",
        filter=models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.doc_id",  # Note the "metadata." prefix!
                match=models.MatchValue(
                    value=doc_id
                    ),
                ),
            ]
        ),
    )


    return results
