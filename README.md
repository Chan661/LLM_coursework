# LLM Labs: Memory, Storage & API Serving

This repository contains two Jupyter notebook labs exploring practical applications of Large Language Models using LangChain, vector databases, FastAPI, and Gradio.

---

## Lab 4: Adding Memory and Storage to LLMs (`lab4_memory.ipynb`)

### Overview
This lab builds a **vector store QA application** from scratch. It covers document loading, text splitting, embedding generation, vector database storage, and retrieval-augmented question answering (RAG) — with conversational memory.

### Key Concepts
- **PDF Loading** — Comparing LangChain PDF loaders (`PyPDFLoader`, `UnstructuredPDFLoader`, `PDFMinerLoader`) for different document types
- **Text Splitting** — Chunking documents using `RecursiveCharacterTextSplitter` for effective retrieval
- **Embeddings & Vector Store** — Generating embeddings with `SentenceTransformer` (MiniLM) and storing them in ChromaDB or Pinecone
- **QA Chains** — Using LangChain's `load_qa_chain` with chain types: `stuff`, `map_reduce`
- **Conversational Memory** — Adding `ConversationBufferMemory` to maintain multi-turn dialogue context
- **Agents with Tools** — Building a LangChain agent that routes questions to the appropriate QA system (e.g., Harry Potter vs. research papers)

### Main Task
Build a **conversational research assistant** over a collection of academic papers. The system answers questions about specific methods and results with responses under 100 words, backed by ChromaDB embeddings.

### Setup

#### Prerequisites
- Python 3.8+
- API keys for **OpenAI**, **SerpAPI**, and **Pinecone**

#### Installation
```bash
pip install -r lab4_requirements.txt
```

#### Environment Variables
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your-openai-api-key
SERPAPI_API_KEY=your-serpapi-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENV=your-pinecone-environment
```

> **Note:** The Pinecone index must be created with **dimension 1536** (matching OpenAI `text-embedding-ada-002` output).

#### Dependencies (`lab4_requirements.txt`)
Key packages include:
- `langchain==0.1.13`, `langchain-openai==0.1.0`, `langchain-community==0.0.29`
- `chromadb==0.4.10` — local vector store
- `pinecone-client==2.2.4` — managed vector store
- `sentence-transformers==2.2.2` — local embedding models
- `pypdf==3.17.0`, `unstructured==0.5.6` — PDF loaders
- `openai==1.3.6`, `tiktoken==0.5.1`

---

## Lab 7: LLM API Server and Web Interfaces (`lab7.ipynb`)

### Overview
This lab teaches how to **serve LLM models as web services** and build interactive frontends — without any frontend language knowledge. It covers HTTP APIs, FastAPI server development, and Gradio UI creation.

### Key Concepts
- **HTTP Requests** — Using Python's `requests` library for GET and POST calls; JSON data exchange
- **FastAPI Server** — Building a REST API that wraps a local LLM (Phi-3) using `fastapi` + `uvicorn`; defining endpoints like `/run` and `/chat`
- **Client-Side Chat History** — Storing and passing conversation history from the client to maintain context across turns
- **Gradio UI** — Creating interactive ML demos with `gradio`; building a chat interface that calls the FastAPI backend
- **Decoupled Architecture** — Separating the model server (FastAPI) from the UI server (Gradio) for scalability
- **Multi-modal Extension** — Serving image generation models (Stable Diffusion) and speech models (SpeechT5, Whisper) via Gradio

### Architecture
```
User Browser
    │
    ▼
Gradio UI Server (port 7860)
    │  HTTP POST /chat
    ▼
FastAPI LLM Server (port 54223)
    │
    ▼
Local LLM (e.g., Phi-3 via HuggingFace Transformers)
```

### Setup

#### Prerequisites
- Python 3.8+
- A local LLM model (e.g., Phi-3); the lab uses models from `/share/model/`

#### Installation
```bash
pip install requests fastapi uvicorn websockets gradio gradio-client
```

#### Running the Services

1. **Start the LLM API server:**
   ```bash
   python /tmp/llm_api.py
   ```

2. **Start the Gradio UI** (in a separate terminal):
   ```bash
   python /tmp/chatUI.py
   ```

3. **Access the interface** at `http://localhost:7860` (or the next available port if 7860 is occupied).

#### Example API Call
```python
import requests, json

query = "What are interesting places to visit?"
history = [{"role": "user", "content": "What is the capital of China?"}]
data = {"query": query, "history": history}

response = requests.post("http://localhost:54223/run", data=json.dumps(data).encode("utf-8"))
print(response.content.decode(response.encoding))
```

---

## Repository Structure

```
.
├── lab4_memory.ipynb        # Vector store QA & conversational memory lab
├── lab7.ipynb               # FastAPI LLM server & Gradio web UI lab
├── lab4_requirements.txt    # Python dependencies for Lab 4
└── README.md
```

---

## References

- [LangChain Documentation](https://python.langchain.com/docs/)
- [Ask A Book Questions (LangChain Tutorial)](https://github.com/gkamradt/langchain-tutorials)
- [ChromaDB](https://docs.trychroma.com/)
- [Pinecone](https://docs.pinecone.io/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Gradio](https://www.gradio.app/docs/)
- [SerpAPI](https://serpapi.com/)
