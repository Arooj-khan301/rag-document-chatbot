# Document-Based Retrieval-Augmented Generation (RAG) Pipeline

**Repository:** `rag-document-chatbot`  
**Author:** Arooj Khan (MS in Computer Science, The University of Haripur)

## 1. Overview

This repository contains the document ingestion, recursive text chunking, dense vector embedding, and FAISS similarity search pipeline developed during my **Prompt Engineering & RAG work** (`PDF_Chatbot_RAG.ipynb`). It enables semantic retrieval over multi-page academic and technical PDF documents using **LangChain**, **Hugging Face Sentence-Transformers**, and **FAISS**.

## 2. Implemented Architecture (Verified in Code)

1. **PDF Ingestion (`PyPDFLoader`):** Extracts page-level text and structural metadata (`source`, `total_pages`, `page`, `page_label`) from multi-page PDF documents (tested in notebook outputs on 50-page and 64-page university lecture slide decks).
2. **Overlapping Text Chunking (`RecursiveCharacterTextSplitter`):** Splits extracted pages into manageable semantic chunks (`chunk_size = 500`, `chunk_overlap = 100`), producing 80 overlapping chunks from a 64-page lecture deck.
3. **Dense Vector Embeddings (`HuggingFaceEmbeddings`):** Encodes document chunks and user queries into 384-dimensional dense vectors using `sentence-transformers/all-MiniLM-L6-v2`.
4. **Vector Indexing & Persistence (`FAISS`):** Builds an in-memory FAISS vector index (`FAISS.from_documents`), persists the index to disk (`vectorstore.save_local("faiss_index")`), and executes top-$k$ ($k=3$) semantic similarity search (`vectorstore.similarity_search(query, k=3)`).

> **Implementation Note:** The current notebook and CLI module implement the complete **ingestion, chunking, embedding, FAISS indexing, and top-$k$ retrieval** stages. Although `langchain-groq` is included in dependencies, the downstream LLM answer-synthesis step (`ChatGroq` prompt chain) is left modular so users can plug in their preferred LLM provider via `.env`.

## 3. Repository Structure

```text
rag-document-chatbot/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── notebooks/
│   └── PDF_Chatbot_RAG.ipynb
├── src/
│   ├── __init__.py
│   └── rag_pipeline.py
└── data/
```

## 4. Installation & Usage

```bash
git clone https://github.com/Arooj-khan301/rag-document-chatbot.git
cd rag-document-chatbot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run PDF ingestion, FAISS indexing, and top-k semantic retrieval
python3 -m src.rag_pipeline \
  --pdf-path data/sample_lecture.pdf \
  --query "What is Generative AI?" \
  --chunk-size 500 \
  --chunk-overlap 100 \
  --top-k 3
```
