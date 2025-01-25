import os

from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from PyPDF2 import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(url="http://localhost:6333")


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def create_docs(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    documents = []
    for chunk in chunks:
        documents.append(Document(page_content=chunk))
    return documents


def ingest_in_vectordb(pdf_path, username):
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")
    client.create_collection(
        collection_name=username,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    vectorstore = Qdrant(client=client, collection_name=username, embeddings=embeddings)
    pdf_text = extract_text_from_pdf(pdf_path)
    documents = create_docs(pdf_text)
    try:
        vectorstore.add_documents(documents)
        print(
            f"Ingested {len(documents)} documents into Qdrant collection '{username}'."
        )
    except Exception as e:
        print(e)
