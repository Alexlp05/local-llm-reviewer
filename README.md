# Local LLM Multi-Agent Literature Reviewer

An in-browser AI agent capable of reading, understanding, and discussing research papers using local LLMs (WebLLM) and RAG (Retrieval-Augmented Generation).

## 1. Theory vs. Practice: From Python to Web

This project bridges the gap between fundamental Deep Learning concepts (RNNs, Transformers) and modern Web-based AI implementation.

The table below maps the theoretical components found in the provided Python scripts (`rnn.py`, `transformer.py`) to their practical counterparts in this web application (`WebLLM`, `Transformers.js`).

| Concept | Theory (Python Implementation) | Practice (Web Technologies) |
| :--- | :--- | :--- |
| **Sequential Processing** | **RNN (`rnn.py`)**: Defines a loop `for i in range(t.size(1))` to process tokens one by one, updating a hidden state `h_state` at each step. This represents the "memory" of the sequence. | **LLM Generation**: Although Transformers process prompts in parallel, text *generation* is sequential (autoregressive). The "hidden state" is effectively replaced by the **Key-Value (KV) Cache** in WebLLM to avoid recomputing past tokens. |
| **Attention Mechanism** | **Bahdanau Attention**: Weights input parts based on relevance to the current decoder state. <br> **Self-Attention (`transformer.py`)**: `CausalSelfAttention` class calculates `q, k, v` vectors and uses `scaled_dot_product_attention` to let tokens "look at" each other. | **WebGPU Kernels**: The massive matrix multiplications required for Self-Attention (Q @ K^T) are executed in parallel on the user's GPU using **WebGPU** compute shaders (via WebLLM/TVM), allowing real-time inference in the browser. |
| **Embeddings** | **`nn.Embedding`**: Converts integer token IDs into dense vectors (`t_emb` in `rnn.py`/`transformer.py`). Captures semantic meaning in a high-dimensional space. | **Transformers.js**: We use a specific model (`Xenova/all-MiniLM-L6-v2`) to generate embeddings for our RAG system. These vectors are stored in our local "Vector Store" (JSON) to enable semantic search over PDF chunks. |
| **Positional Encoding** | **`p_emb` (`transformer.py`)**: Adds a learnable vector (`nn.Embedding`) to the token embedding to give the model a sense of order/position in the sequence. | **Rotary Positional Embeddings (RoPE)**: Modern LLMs (like Llama-3 used here) use relative positional encodings (RoPE) baked into the model architecture loaded by WebLLM. |
| **Context Window** | **`ctx_size`**: Limiting the input to the last N tokens (`idxs[:, -ctx_size:]` in generation loop). | **Context Limit**: WebLLM manages a finite context window (e.g., 2048 or 8192 tokens). Our RAG system acts as an "external memory" to retrieve relevant information that fits into this window. |

## Project Roadmap

- [x] **Step 1: Infrastructure & Theory**: Setup Repo, HTML/CSS, and Theory comparison.
- [x] **Step 2: The Brain**: WebLLM integration for local Chat.
- [x] **Step 3: The Eyes**: PDF parsing and Chunking.
- [x] **Step 4: The Memory**: Embedding generation with Transformers.js.
- [x] **Step 5: The Reasoning**: RAG engine (Vector Search + System Prompt).
- [x] **Step 6: Bonus**: TTS, Dark Mode, Voice Input (STT), Citation UI.

## Getting Started

1. Open `index.html` in a modern browser (Chrome/Edge/Brave).
2. Ensure you have a GPU compatible with WebGPU.

## Features (Phase 2 Upgrade)

- **System Controls**: Adjust Temperature and System Prompts in real-time.
- **Citation Badges**: See exactly which PDF source was used for the answer.
- **Voice Mode**: Speak to the agent using the Microphone button (Whisper-tiny model).
