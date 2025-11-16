# Multi-Document Hybrid Search RAG System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-white?style=for-the-badge&logo=langchain&logoColor=black)](https://www.langchain.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-white?style=for-the-badge&logo=fastapi&logoColor=green)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-blue?style=for-the-badge&logo=docker)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-EC2-orange?style=for-the-badge&logo=amazonaws)](https://aws.amazon.com/ec2/)

This project is an advanced, agentic RAG (Retrieval-Augmented Generation) system designed to answer complex queries across multiple PDF documents. It integrates a sophisticated Hybrid Search mechanism and a state-of-the-art re-ranking model to deliver highly accurate, low-hallucination answers.

The entire RAG pipeline is exposed as a scalable FastAPI microservice, containerized with Docker for easy deployment on platforms like AWS EC2.

## Problem Statement

Standard vector-only RAG systems often struggle with:
1.  **Keyword Mismatch:** Dense retrievers (like FAISS) can miss relevant documents that don't share semantic meaning but share exact keywords (e.g., "HTTP/2").
2.  **Lack of Precision:** Sparse retrievers (like BM25) are great with keywords but lack semantic understanding.
3.  **Context Noise:** Feeding low-quality or irrelevant chunks to the LLM is a primary cause of hallucination and incorrect answers.

This project solves these problems by creating a multi-stage pipeline that gets the best of both worlds.

## Key Features

* **Agentic Framework:** Developed a **LangChain** agent with custom **RetrievalTools** to intelligently query multiple PDF documents simultaneously.
* **Hybrid Search:** Engineered a **Hybrid Search** mechanism that combines the strengths of sparse retrieval (**BM25**) and dense retrieval (**FAISS**) to improve retrieval accuracy.
* **Advanced Re-ranking:** Integrated a **Cohere Re-ranker** to re-order the retrieved context, prioritizing the most relevant information and significantly **reducing LLM hallucination**.
* **Scalable Deployment:** Deployed the RAG agent as a **FastAPI** microservice, containerized with **Docker** and suitable for deployment on **AWS EC2**.

## How It Works: The RAG Pipeline

1.  **Ingestion:** PDFs are loaded, processed, and split into text chunks.
2.  **Indexing:** Two separate indexes are created for each document:
    * A dense **FAISS** vector store for semantic search.
    * A sparse **BM25** index for keyword-based search.
3.  **Query:** A user sends a query to the FastAPI endpoint.
4.  **Retrieval (Hybrid):** The LangChain agent receives the query and its custom tools retrieve documents from *both* the FAISS and BM25 indexes. The results are combined.
5.  **Re-ranking:** The (potentially noisy) combined list of retrieved chunks is passed to the **Cohere Re-ranker**. The re-ranker filters and re-orders the chunks, pushing the most relevant ones to the top.
6.  **Generation:** The top-k, high-quality, re-ranked chunks are passed to the LLM (e.g., GPT-4) as context.
7.  **Response:** The LLM generates a final, accurate answer based on the curated context, which is streamed back to the user.

## Tech Stack

* **Core:** Python
* **LLM Framework:** LangChain (Agents, Custom Tools)
* **Retrieval:**
    * Hybrid Search (BM25 + FAISS)
    * Cohere Re-ranker
* **API & Deployment:** FastAPI, Docker, AWS EC2
* **Vector Store:** FAISS (or ChromaDB/Weaviate)

## Getting Started

### Prerequisites

* Python 3.10+
* Docker & Docker Compose
* API keys for:
    * OpenAI (or your chosen LLM provider)
    * Cohere (for the re-ranker)

### Installation & Local Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/THE-AkS-21/hybrid-search-rag.git
    cd hybrid-search-rag
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    # On Windows:
    venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root directory and add your API keys:
    ```.env
    OPENAI_API_KEY="your_openai_key_here"
    COHERE_API_KEY="your_cohere_key_here"
    ```

### Running with Docker (Recommended)

This is the simplest way to get the service running.

```bash
docker-compose up --build
