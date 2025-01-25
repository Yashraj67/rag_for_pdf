from flashrank import Ranker, RerankRequest
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")


def rerank(query, docs):
    ranker = Ranker(
        model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/home/yashraj/backend/"
    )
    raw_docs = []
    i = 1
    for doc in docs:
        raw_docs.append({"id": i, "text": doc.page_content})
        i = i + 1

    rerankrequest = RerankRequest(query=query, passages=raw_docs)
    results = ranker.rerank(rerankrequest)
    ranked_docs = []
    for doc in results:
        if doc["score"] > 0.001:
            ranked_docs.append(doc["text"])

    return ranked_docs


def get_relevant_docs(question, collection_name):
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 25})
    relevant_docs = retriever.invoke(question)
    reranked_docs = rerank(question, relevant_docs)
    return reranked_docs
