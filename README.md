All work is completed through remote connection of K8S clusters using VS Code.

### Lab 4: Build a conversation system from a collection of research papers.
In this lab, I build a conversation system from a collection of research papers I selected via ChromaDB, LangChain, and OpenAI. I load and create a vector embedding for the documents. It supports content mapping and matching through similarity search of vector embedding, and question answering through chaining with various LangChain's question-answer chain that deploys large language model such as GPT-3.5 turbo.

### Lab 7: Build a web UI for LLM through an LLM API server.
The better way for users to provide online LLM service is to call API server using request, instead of running the LLM within user interface (UI) server process. With this, I employ Gradio as my UI and FastAPI for my local LLM. The LLM API server uses GET and POST handler to receive the user's message and return response. The user can send request to Stable Diffusion through online gradio port and get the generated image.
