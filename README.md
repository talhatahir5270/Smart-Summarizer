# 🧠 Text Summarizer and Chatbot

A powerful **Streamlit-based NLP application** that allows users to **upload Word documents**, generate intelligent summaries, and interact with a **chatbot** that answers questions based on the document's content. It combines extractive and abstractive summarization methods with a query-driven chatbot for enhanced document understanding.


## 🔍 Key Features

- 📄 **Document Upload**: Upload `.docx` files and preview their content.
- ✂️ **Text Summarization**:
  - **TF-IDF Extractive Summary**
  - **Embedding-based Summary**
  - **Abstractive Summary** (via BART transformer)
- 🤖 **Chatbot Interaction**:
  - Ask questions based on the uploaded document.
  - Real-time conversation history tracking.
- 📚 **Semantic Search** using ChromaDB and vector embeddings.

---

## 🧠 Core Modules

### 1. **Summarization Module**
- **TF-IDF Method**: Scores and extracts key sentences based on word importance.
- **Embedding-Based Method**: Uses sentence embeddings and cosine similarity.
- **Abstractive Method**: Generates human-like summaries using BART transformer.

### 2. **Chatbot Module**
- Embeds text chunks into a vector database (ChromaDB).
- Accepts user queries and returns context-relevant answers.
- Employs **LLM-based response generation** using retrieved document chunks.

### 3. **Embedding & Chunking**
- Large documents are recursively chunked.
- Embeddings are generated via HuggingFace (e.g., BAAI/bge-large-en-v1.5).
- Chunks are stored/retrieved using cosine similarity search in ChromaDB.

---

## 🧰 Tech Stack

- **Frontend**: Streamlit  
- **Backend**: Python  
- **NLP Libraries**: NLTK, SentenceTransformers, Transformers  
- **LLM**: LLaMA 3 via Ollama  
- **Vector DB**: ChromaDB  
- **Document Parsing**: `python-docx`  
- **Model**: BART for abstractive summarization  

