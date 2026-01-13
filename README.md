# Local LLM Multi-Agent Literature Reviewer

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![WebGPU](https://img.shields.io/badge/Powered%20by-WebGPU-green)
![Privacy](https://img.shields.io/badge/Privacy-100%25%20Local-purple)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Launch%20App-red)](https://alexlp05.github.io/local-llm-reviewer/)

**A professional, in-browser AI agent for analyzing research papers.**
This private application runs entirely on your device using WebGPU, allowing you to chat with documents, synthesize literature reviews, and explore complex topics without sending data to the cloud.

![Application Interface Placeholder](./interface_screenshot.png)
*> Replace this image with a screenshot of your active interface showing a Literature Review*

---

## 🚀 Key Features

### 🧠 Fully Local Intelligence

- **WebLLM Integration**: Runs Llama-3-8B (or smaller variants) directly in your browser.
- **RAG Engine**: "Retrieval-Augmented Generation" allows the AI to read your PDFs and cite specific sources.
- **100% Privacy**: No data ever leaves your computer. Your research papers stay safe.

### 📚 Advanced Literature Review

- **Smart Synthesis**: Just upload 6+ PDFs and ask for a review. The agent compares and contrasts themes.
- **Structure**: Automatically formats outputs with Introductions, Method Comparisons, and Conclusions.
- **Citations**: Returns precise `[source: filename]` citations for every claim.

### 🎭 AI Personas

Instantly switch the AI's behavior to match your needs:

- **🎓 The Scholar**: Formal, academic, structured synthesis (Default).
- **💡 The Simplifier**: Explains complex concepts with analogies and bullet points (ELI5).
- **🕵️ The Critic**: Identifies methodological flaws and contradictions.

### 🎙️ Accessibility & UI

- **Voice Mode**: Speak to the agent (STT) and hear responses (TTS).
- **Modern Thread View**: A clean, professional chat interface (like ChatGPT/Claude).
- **Dark Mode**: Optimized for late-night research sessions.

---

## 🛠️ Technology Stack

This project bridges the gap between Python theory (`rnn.py`) and Web practice:

| Component | Library / Tech | Description |
| :--- | :--- | :--- |
| **Model Runtime** | **WebLLM** (TVM) | Hardware-accelerated LLM inference via WebGPU. |
| **Embeddings** | **Transformers.js** | Generates vector embeddings for RAG (`Xenova/all-MiniLM-L6-v2`). |
| **PDF Parsing** | **PDF.js** | Extracts text from uploaded research papers. |
| **Styling** | **Tailwind CSS** | Modern utility-first CSS for a responsive design. |
| **Language** | **Vanilla JS (ES6+)** | Lightweight, no complex build steps required. |

---

## 💻 Getting Started

### Prerequisites

- A modern browser (Chrome 113+, Edge) with **WebGPU** support.
- A dedicated GPU (NVIDIA/AMD) is recommended for 8B models.

### Local Setup

Since this app uses WebGPU and Modules, it must be served over HTTP (not `file://`).

1. **Clone the Repository**

    ```bash
    git clone https://github.com/Alexlp05/local-llm-reviewer.git
    cd local-llm-reviewer
    ```

2. **Start a Local Server**
    You can use any static file server.

    *Option A: Node.js (Recommended)*

    ```bash
    npx serve .
    ```

    *Option B: Python*

    ```bash
    python -m http.server 8000
    ```

3. **Launch**
    - Open `http://localhost:3000` (or the port shown in terminal).
    - **First Run**: The app will download the AI model (~4GB) to your browser cache. This happens only once.

---

## 🌐 Deployment

This project is deployed and live at:
**[https://alexlp05.github.io/local-llm-reviewer/](https://alexlp05.github.io/local-llm-reviewer/)**

To update the deployment:

1. Push your latest changes to the `main` branch.
2. GitHub Pages will automatically rebuild and serve the new version.

---

## ⚖️ License

This project is open-source and available under the MIT License.
