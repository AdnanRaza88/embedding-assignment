# Embedding Assignment - RAG PDF Chatbot

A Streamlit-based RAG (Retrieval-Augmented Generation) chatbot that lets you upload any PDF, embed its content using HuggingFace sentence-transformers, store vectors in Pinecone, and chat with the document using Groq LLM.

## Project Structure

```
rag-pdf-chatbot/
── app.py
── requirements.txt
── .env.example
```

## Features

- PDF upload support
- Text chunking with RecursiveCharacterTextSplitter
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- Vector store: Pinecone (with unique namespace per PDF)
- LLM: Groq (`llama3-70b-8192`)
- Clean Streamlit chat interface

## Setup

1. Clone the repo and go to the project folder:

```bash
cd rag-pdf-chatbot
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

Required keys:
- `PINECONE_API_KEY`
- `GROQ_API_KEY`

Optional:
- `HUGGINGFACE_API_TOKEN` (only if using a gated model)

4. Run the app:

```bash
streamlit run app.py
```

## How it works

1. Upload a PDF
2. The PDF is loaded, split into chunks, and embedded
3. Embeddings are stored in Pinecone under a unique namespace (based on filename + content hash)
4. Ask questions — the system retrieves relevant chunks and generates answers using Groq

## Daily Activity Log

- **1 Aug 2026**: README updated as part of daily GitHub commit streak + badges practice.
