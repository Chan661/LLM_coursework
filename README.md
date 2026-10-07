# LLM Coursework: Two Stage Projects from a Ten-Lab Course

Coursework from the Tsinglan S-Plan program (Nov 2023 – Mar 2024): a course of ten hands-on labs on building LLM applications, with materials designed by Prof. Wei Xu (Tsinghua University). Each lab sets a problem and gives starter code. This repository holds my completed work for two of them.

Each lab works as a small standalone project, and together they cover the two halves of an LLM application: giving a model access to knowledge (Lab 4) and serving a model to users (Lab 7).

Instructions and starter code come from the course. My work is in the "Your Task" cells and the experiments around them.

## At a glance

| Lab | Project | Stack | Outcome |
|---|---|---|---|
| 4 | Question answering over a research paper | LangChain, SentenceTransformers (MiniLM), ChromaDB, OpenAI | Answers questions on a paper's topic and results from 135 indexed chunks |
| 7 | Serving an LLM and an image model behind a web UI | FastAPI, uvicorn, Gradio, Hugging Face Transformers, Stable Diffusion | Phi-3 chat served over HTTP with a Gradio UI; Stable Diffusion served the same way |

---

## Lab 4: A research-paper assistant (`Lab4/lab4_memory.ipynb`)

**Problem.** A language model knows nothing about a specific paper. The assignment was to build a conversational system that answers specific questions about the methods and results of research papers of my choice, backed by a persistent vector database, with example queries and a written summary.

**What I built.** A retrieval-augmented question-answering pipeline over one paper (the BARTScore paper):

1. **Load** the PDF with LangChain's `PyPDFLoader`.
2. **Split** it with `RecursiveCharacterTextSplitter` (`chunk_size=500`, `chunk_overlap=20`), giving 135 chunks.
3. **Embed** each chunk with the local `all-MiniLM-L12-v2` SentenceTransformer model.
4. **Store** the embeddings in a persistent ChromaDB index.
5. **Retrieve and answer.** `similarity_search` finds the closest chunks, then `load_qa_chain` answers with OpenAI's `gpt-3.5-turbo-instruct` (temperature 0).
6. **Compare chain types.** Three example queries cover plain similarity search, a `stuff` chain, and a `map_reduce` chain.

**Findings.**

- **Worked:** questions about the paper's topic and its experimental results were answered from the retrieved chunks.
- **Phrasing mattered:** a vague query such as "what is the paper's experimental result?" failed until I reworded it in more detail. A query for the paper's authors returned a conclusion passage instead of the author list.
- **Memory:** follow-up questions have no context on their own, which is why I started on conversational memory. `ConversationBufferMemory` is set up in the notebook, but I did not finish connecting it to the QA chain.
- **Scope:** one paper is indexed, answers are not length-limited, and the lab's multi-source agent example is included but was not run. Pinecone and SerpAPI appear in the lab's setup instructions but are not used in my code.

**What I learned.**

- Retrieval quality depends on how text is chunked and how the question is worded.
- `stuff` and `map_reduce` trade cost against how much text they can handle.
- A QA system needs memory or query rewriting before follow-up questions work.

---

## Lab 7: Serving an LLM with FastAPI and Gradio (`lab7.ipynb`)

**Problem.** Running a model inside a notebook is not a service. The assignment was to build my own API server for a local LLM and put a web interface on it, without writing any front-end code.

**What I built.**

1. A **GET `/run/` endpoint** that takes a prompt string, decodes it, and returns the model's reply.
2. **POST `/run/` and `/chat` endpoints** that accept a `query` and a `history` (validated with a Pydantic model). The client keeps the conversation history and sends it with each request, so the server stays stateless.
3. The **server entry point:** loads Phi-3-mini-128k-instruct onto the GPU with Hugging Face Transformers and starts `uvicorn` on port 54223.
4. A **Python client test** using `requests` that sends a follow-up question with history and gets a status 200 reply.
5. A **Gradio chat UI** (`chatUI.py`) that calls the `/chat` endpoint over HTTP, keeping the model server separate from the UI server.
6. An **extension to image generation.** A `/generate_image` endpoint serves a Stable Diffusion model (from Lab 5) and returns the image as base64, a second Gradio UI displays it, and a test through `gradio_client` returned a 512×512 image. I debugged this endpoint with logging.

```
User Browser
    │
    ▼
Gradio UI server (port 7860)
    │  HTTP POST /chat
    ▼
FastAPI server (port 54223)
    │
    ▼
Local model (Phi-3-mini via Hugging Face Transformers)
```

**Limitations.** The image endpoint reloads the Stable Diffusion pipeline on every request. Loading it once at startup would be the fix. Chat history also grows with every request, since the client resends all of it.

**What I learned.**

- How to design a stateless model server and keep the UI a thin client.
- Why history and long prompts go in a POST body instead of the URL.
- How to debug a service with logging when requests never reach a handler.

---

## How the two labs fit together

Lab 4 is the knowledge side of an LLM application (retrieval over documents). Lab 7 is the delivery side (a model behind an API and a UI). They are not combined in this repository. Serving the Lab 4 pipeline behind the Lab 7 server is the natural next step.

---

## Running the notebooks

The notebooks were run on the course GPU cluster, and several paths point to it (`/share/...`, `/ssdshare/...`). To rerun elsewhere, replace them with local paths or Hugging Face model IDs (for example `sentence-transformers/all-MiniLM-L12-v2` and `microsoft/Phi-3-mini-128k-instruct`).

**Lab 4**

```
pip install -r Lab4/lab4_requirements.txt
```

Create a `.env` file with `OPENAI_API_KEY=your-key`. Never commit this file.

**Lab 7** (a GPU is needed for the models)

```
pip install requests fastapi uvicorn gradio gradio-client transformers diffusers torch
```

The notebook writes the two server scripts to `/tmp`. Start them in separate terminals:

```
python /tmp/llm_api.py
python /tmp/chatUI.py
```

Then open `http://localhost:7860`.

---

## Repository structure

```
.
├── Lab4/
│   ├── lab4_memory.ipynb        # Research-paper QA pipeline
│   └── lab4_requirements.txt    # Python dependencies for Lab 4
├── lab7.ipynb                   # FastAPI LLM server and Gradio UI
└── README.md
```

---

## Acknowledgements

Lab instructions and materials were designed by Prof. Wei Xu at Tsinghua University.

## References

- [LangChain Documentation](https://python.langchain.com/docs/)
- [Ask A Book Questions (LangChain Tutorial)](https://github.com/gkamradt/langchain-tutorials)
- [ChromaDB](https://docs.trychroma.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Gradio](https://www.gradio.app/docs/)
