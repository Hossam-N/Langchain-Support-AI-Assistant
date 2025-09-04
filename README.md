# **RAG Assistant**

**RAG Assistant** is an AI-powered tool that enhances information retrieval and question answering using Retrieval-Augmented Generation (RAG) techniques with open-source LLMs.  
It scrapes documentation, preprocesses text, builds embeddings with ChromaDB, and serves responses via an interactive Gradio app.

---

## **Features**
- **Web Scraping:** Collect technical documentation (e.g., LangChain docs).
- **Preprocessing:** Clean and chunk documents for embedding.
- **Semantic Search:** Retrieve relevant context using MiniLM embeddings + ChromaDB.
- **Contextual Answers:** Generate accurate responses with Flan-T5 (Hugging Face).
- **Interactive Demo:** Gradio app for user-friendly Q&A.
- **Extensible:** Easily swap datasets, embeddings, or models.

---

## **Requirements**
- **Python 3.9+**
- **PyTorch**
- **Transformers**
- **Sentence-Transformers**
- **LangChain**
- **ChromaDB**
- **BeautifulSoup4** + **Requests**
- **Gradio**

_All dependencies are listed in `requirements.txt`._

---

## **Installation**
1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/rag-assistant.git
    cd rag-assistant
    ```
2. **Create a virtual environment and activate it:**
    - Mac/Linux:
      ```bash
      python -m venv venv
      source venv/bin/activate
      ```
    - Windows:
      ```cmd
      python -m venv venv
      venv\Scripts\activate
      ```
3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## **Data Preparation**
The `data/` folder is not included in this repository (to keep it lightweight).  
You can regenerate it using the scripts provided:

1. **Scrape Docs:**
    ```bash
    python -m src.ingestion.scrape_docs --output data/raw/docs.json
    ```
2. **Preprocess:**
    ```bash
    python -m src.ingestion.preprocess --input data/raw/docs.json --output data/processed/chunks.json
    ```
3. **Build Vectorstore:**
    ```bash
    python -m src.rag.pipeline --input data/processed/chunks.json --output data/vectorstore
    ```

---

## **Usage**
1. **Launch the Gradio app:**
    ```bash
    python -m src.app.gradio_app
    ```
2. **Open** [http://localhost:7860](http://localhost:7860) **in your browser.**

---

## **Tech Stack**
- **Python**
- **LangChain** (retrievers, pipelines)
- **Hugging Face Transformers** (Flan-T5, MiniLM)
- **ChromaDB** (vector database)
- **Gradio** (demo interface)
- **BeautifulSoup4** & **Requests** (scraping)

---

## **License**
This project is licensed under the **MIT License**.

---
