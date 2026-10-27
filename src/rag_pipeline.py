"""Document ingestion, recursive chunking, dense embedding, and FAISS retrieval.

Refactored directly from `PDF_Chatbot_RAG.ipynb`.
"""

import argparse
import os
from typing import List


def build_and_query_faiss_index(
    pdf_path: str,
    query: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    index_Output_dir: str = "faiss_index",
    top_k: int = 3,
) -> List[object]:
    """Loads a PDF, splits into overlapping chunks, builds a FAISS index, and retrieves top-k chunks."""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embedding_model)
    vectorstore.save_local(index_Output_dir)

    results = vectorstore.similarity_search(query, k=top_k)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF RAG Ingestion & Retrieval CLI")
    parser.add_argument("--pdf-path", required=True, help="Path to input PDF file")
    parser.add_argument("--query", default="What is Generative AI?", help="Search query")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    retrieved_docs = build_and_query_faiss_index(
        pdf_path=args.pdf_path,
        query=args.query,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
    )
    for idx, doc in enumerate(retrieved_docs, start=1):
        print("=" * 80)
        print(f"Result {idx} | Metadata: {doc.metadata}")
        print("=" * 80)
        print(doc.page_content[:500])
