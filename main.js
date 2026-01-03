import * as webllm from "https://esm.run/@mlc-ai/web-llm";

/****************************************************************************
 * CONFIGURATION
 ****************************************************************************/
const SELECTED_MODEL = "Llama-3-8B-Instruct-q4f32_1-MLC";

// DOM Elements
const dom = {
    chatBox: document.getElementById('chat-box'),
    userInput: document.getElementById('user-input'),
    sendBtn: document.getElementById('send-btn'),
    statusBar: document.getElementById('status-bar'),
    statusText: document.getElementById('status-text'),
    progressBarContainer: document.querySelector('.progress-bar-container'),
    progressBar: document.getElementById('progress-bar'),
    dropZone: document.getElementById('drop-zone'),
    fileInput: document.getElementById('file-input')
};

let engine = null;
let vectorStore = []; // To store chunks { id, text, vector }

/****************************************************************************
 * INITIALIZATION
 ****************************************************************************/
async function initializeEngine() {
    updateStatus("Initializing WebLLM...", 0);

    const initProgressCallback = (report) => {
        const label = report.text || report;
        updateStatus(label, report.progress || 0);
    };

    try {
        updateStatus("Loading Model (this may take a while)...", 0.01);

        engine = await webllm.CreateMLCEngine(
            SELECTED_MODEL,
            { initProgressCallback: initProgressCallback }
        );

        updateStatus("Ready to chat!", 1);
        enableInput(true);
        addSystemMessage("Model loaded successfully. Say hello!");

    } catch (err) {
        console.error("Initialization error:", err);
        updateStatus("Error loading model: " + err.message, 0);
        addSystemMessage("Error: " + err.message);
    }
}

/****************************************************************************
 * PDF HANDLING & CHUNKING
 ****************************************************************************/
async function handleFile(file) {
    if (file.type !== 'application/pdf') {
        alert("Please upload a PDF file.");
        return;
    }

    addSystemMessage(`Reading ${file.name}...`);
    updateStatus("Reading PDF...", 0.1);

    try {
        const text = await extractTextFromPDF(file);
        addSystemMessage(`PDF Loaded. Length: ${text.length} characters.`);

        // Chunking
        const chunks = chunkText(text, 500, 50); // 500 chars, 50 overlap

        // Store chunks (Preparation for Step 4)
        vectorStore = chunks.map((chunk, index) => ({
            id: index,
            text: chunk,
            vector: null // To be filled in Step 4
        }));

        console.log("Created Chunks:", vectorStore);
        addSystemMessage(`Successfully created ${chunks.length} text chunks. (Check Console)`);
        updateStatus("Ready to chat!", 1);

    } catch (err) {
        console.error("PDF Error:", err);
        addSystemMessage("Error reading PDF: " + err.message);
        updateStatus("Error reading PDF", 0);
    }
}

async function extractTextFromPDF(file) {
    const arrayBuffer = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
    let fullText = "";

    for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const textContent = await page.getTextContent();
        const pageText = textContent.items.map(item => item.str).join(" ");
        fullText += pageText + "\n";

        // Update progress
        updateStatus(`Reading PDF (Page ${i}/${pdf.numPages})...`, i / pdf.numPages);
    }

    return fullText;
}

function chunkText(text, chunkSize = 500, overlap = 50) {
    const chunks = [];
    let start = 0;

    while (start < text.length) {
        const end = Math.min(start + chunkSize, text.length);
        chunks.push(text.slice(start, end));
        start += chunkSize - overlap;
    }

    return chunks;
}

/****************************************************************************
 * CHAT LOGIC
 ****************************************************************************/
async function handleUserMessage() {
    const text = dom.userInput.value.trim();
    if (!text || !engine) return;

    addUserMessage(text);
    dom.userInput.value = "";
    enableInput(false);

    try {
        if (!window.conversationHistory) {
            window.conversationHistory = [
                { role: "system", content: "You are a helpful academic assistant. Answer questions concisely." }
            ];
        }

        window.conversationHistory.push({ role: "user", content: text });

        const botMessageId = addBotMessage("");
        const botBubble = document.getElementById(botMessageId).querySelector('.bubble');

        const messages = window.conversationHistory;

        const chunks = await engine.chat.completions.create({
            messages,
            stream: true,
            max_tokens: 512,
        });

        let fullReply = "";

        for await (const chunk of chunks) {
            const delta = chunk.choices[0]?.delta?.content || "";
            fullReply += delta;
            botBubble.textContent = fullReply;
            dom.chatBox.scrollTop = dom.chatBox.scrollHeight;
        }

        window.conversationHistory.push({ role: "assistant", content: fullReply });
        enableInput(true);

    } catch (err) {
        console.error("Chat error:", err);
        addSystemMessage("Error generating response: " + err.message);
        enableInput(true);
    }
}

/****************************************************************************
 * UI HELPERS
 ****************************************************************************/
function updateStatus(text, progress = 0) {
    dom.statusText.textContent = text;
    if (progress > 0 && progress < 1) {
        dom.progressBarContainer.classList.remove('hidden');
        dom.progressBar.style.width = `${progress * 100}%`;
    } else if (progress >= 1) {
        dom.progressBarContainer.classList.add('hidden');
        dom.progressBar.style.width = '100%';
    }
}

function enableInput(enabled) {
    dom.userInput.disabled = !enabled;
    dom.sendBtn.disabled = !enabled;
    if (enabled) dom.userInput.focus();
}

function addSystemMessage(text) {
    const div = document.createElement('div');
    div.className = 'message system';
    div.innerHTML = `<div class="bubble">${text}</div>`;
    dom.chatBox.appendChild(div);
    dom.chatBox.scrollTop = dom.chatBox.scrollHeight;
}

function addUserMessage(text) {
    const div = document.createElement('div');
    div.className = 'message user';
    div.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
    dom.chatBox.appendChild(div);
    dom.chatBox.scrollTop = dom.chatBox.scrollHeight;
}

function addBotMessage(initialText) {
    const id = "bot-msg-" + Date.now();
    const div = document.createElement('div');
    div.id = id;
    div.className = 'message bot';
    div.innerHTML = `<div class="bubble">${escapeHtml(initialText)}</div>`;
    dom.chatBox.appendChild(div);
    dom.chatBox.scrollTop = dom.chatBox.scrollHeight;
    return id;
}

function escapeHtml(text) {
    if (!text) return "";
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

/****************************************************************************
 * EVENT LISTENERS
 ****************************************************************************/
dom.sendBtn.addEventListener('click', handleUserMessage);
dom.userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleUserMessage();
});

// Drag & Drop
dom.dropZone.addEventListener('click', () => dom.fileInput.click());
dom.fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

dom.dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dom.dropZone.classList.add('dragover');
});

dom.dropZone.addEventListener('dragleave', () => {
    dom.dropZone.classList.remove('dragover');
});

dom.dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dom.dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

// Start initialization on load
window.addEventListener('DOMContentLoaded', () => {
    initializeEngine();
});
