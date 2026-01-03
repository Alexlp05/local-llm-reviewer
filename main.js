import * as webllm from "https://esm.run/@mlc-ai/web-llm";
import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

/****************************************************************************
 * CONFIGURATION
 ****************************************************************************/
const SELECTED_MODEL = "Llama-3-8B-Instruct-q4f32_1-MLC";
const EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";

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
    fileInput: document.getElementById('file-input'),
    ttsToggle: document.getElementById('tts-toggle'),
    systemPrompt: document.getElementById('system-prompt'),
    tempSlider: document.getElementById('temp-slider'),
    tempValue: document.getElementById('temp-value'),
    docStatus: document.getElementById('document-status'),
    micBtn: document.getElementById('mic-btn')
};

let engine = null;
let extractor = null; // Transformers.js pipeline
let transcriber = null; // Whisper pipeline
let vectorStore = []; // To store chunks { id, text, vector }
const synth = window.speechSynthesis;
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];

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
        // Load WebLLM
        updateStatus("Loading Chat Model (WebLLM)...", 0.01);
        engine = await webllm.CreateMLCEngine(
            SELECTED_MODEL,
            { initProgressCallback: initProgressCallback }
        );

        // Load Transformers.js Embedding Model
        updateStatus("Loading Embedding Model (Transformers.js)...", 0.5);
        extractor = await pipeline('feature-extraction', EMBEDDING_MODEL);

        updateStatus("Ready to chat!", 1);
        enableInput(true);
        addSystemMessage("Model & Embedder loaded. Ready!");

    } catch (err) {
        console.error("Initialization error:", err);
        updateStatus("Error loading models: " + err.message, 0);
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

        // Store chunks
        vectorStore = chunks.map((chunk, index) => ({
            id: index,
            text: chunk,
            vector: null,
            source: file.name
        }));

        dom.docStatus.textContent = `File Loaded: ${file.name} (${chunks.length} chunks)`;


        addSystemMessage(`Created ${chunks.length} chunks. Generating Embeddings...`);
        updateStatus("Generating Embeddings...", 0.2);

        // Generate Embeddings
        await embedChunks();

        console.log("Vector Store:", vectorStore);
        addSystemMessage(`Embeddings generated for ${vectorStore.length} chunks. RAG Ready.`);
        updateStatus("Ready to chat!", 1);

    } catch (err) {
        console.error("PDF Error:", err);
        addSystemMessage("Error reading PDF: " + err.message);
        updateStatus("Error reading PDF", 0);
    }
}

async function embedChunks() {
    if (!extractor) {
        addSystemMessage("Embedding model not loaded yet.");
        return;
    }

    const total = vectorStore.length;
    for (let i = 0; i < total; i++) {
        const chunk = vectorStore[i];
        const output = await extractor(chunk.text, { pooling: 'mean', normalize: true });
        chunk.vector = Array.from(output.data);

        if (i % 10 === 0) {
            updateStatus(`Embedding Chunks (${i + 1}/${total})...`, (i + 1) / total);
        }
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
 * SPEECH TO TEXT (WHISPER)
 ****************************************************************************/
async function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

async function startRecording() {
    if (!navigator.mediaDevices) {
        alert("Microphone not supported.");
        return;
    }

    try {
        // Load transcriber on first use
        if (!transcriber) {
            updateStatus("Loading Whisper Model...", 0.5);
            addSystemMessage("Loading Whisper (STT)...");
            transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');
            addSystemMessage("Whisper Loaded.");
            updateStatus("Ready to Record", 1);
        }

        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            await transcribeAudio(audioBlob);

            // Cleanup stream tracks
            stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.start();
        isRecording = true;
        dom.micBtn.classList.add('recording');
        updateStatus("Listening...", 1);

    } catch (err) {
        console.error("Mic Error:", err);
        addSystemMessage("Mic Error: " + err.message);
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        dom.micBtn.classList.remove('recording');
        updateStatus("Processing Audio...", 1);
    }
}

async function transcribeAudio(audioBlob) {
    if (!transcriber) return;

    try {
        // Convert Blob to URL
        const url = URL.createObjectURL(audioBlob);

        // Transcribe
        // Transformers.js handles URL or AudioContext. Let's try passing the URL directly.
        // Note: Transformers.js usually expects float32 array or URL.
        const output = await transcriber(url);

        const text = output.text;
        if (text) {
            dom.userInput.value = text.trim();
            handleUserMessage(); // Auto-send
        }

    } catch (err) {
        console.error("Transcribe Error:", err);
        addSystemMessage("STT Error: " + err.message);
    }
    updateStatus("Ready to chat!", 1);
}


/****************************************************************************
 * RAG ENGINE & CHAT LOGIC
 ****************************************************************************/
async function handleUserMessage() {
    const text = dom.userInput.value.trim();
    if (!text || !engine) return;

    addUserMessage(text);
    dom.userInput.value = "";
    enableInput(false);

    try {
        let systemPrompt = dom.systemPrompt.value;
        let relevantContext = "";

        // RAG STEP: If we have documents, find relevant chunks
        if (vectorStore.length > 0 && extractor) {
            addSystemMessage("Searching documents...");

            // 1. Embed Query
            const output = await extractor(text, { pooling: 'mean', normalize: true });
            const queryVector = Array.from(output.data);

            // 2. Cosine Similarity Search
            const similarities = vectorStore.map(chunk => ({
                ...chunk,
                score: cosineSimilarity(queryVector, chunk.vector)
            }));

            // 3. Sort & Top-K
            similarities.sort((a, b) => b.score - a.score);
            const topK = similarities.slice(0, 3); // Top 3 chunks

            // 4. Construct Context
            relevantContext = topK.map(chunk => `[Context]: ${chunk.text}`).join("\n\n");
            console.log("RAG Context:", topK);

            systemPrompt += "\n\nCONTEXT:\n" + relevantContext;
        }

        // WebLLM logic
        if (!window.conversationHistory) {
            window.conversationHistory = [];
        }

        // We rebuild messages every time to ensure System Prompt includes latest RAG context if needed,
        // BUT WebLLM session handling usually appends. 
        // Strategy: Just append the user message, but if we did RAG, we might prepend context to the USER message 
        // OR update the system prompt. Llama-3 handles system prompts well.
        // Let's UPDATE the system prompt for THIS turn? Hard with simple history array.
        // Best approach for single-turn RAG chat: 
        // Send: System (with context) + User Query.

        // For multi-turn, we'll just inject context into the User message for simplicity and effectiveness.
        // "Context: ... \n\n Question: ..."

        const finalUserMessage = relevantContext
            ? `Context:\n${relevantContext}\n\nQuestion: ${text}`
            : text;

        const messages = [
            { role: "system", content: "You are a helpful academic assistant." },
            ...window.conversationHistory, // Past history
            { role: "user", content: finalUserMessage }
        ];

        // Stream response
        const botMessageId = addBotMessage("");
        const botBubble = document.getElementById(botMessageId).querySelector('.bubble');

        const chunks = await engine.chat.completions.create({
            messages,
            stream: true,
            temperature: parseFloat(dom.tempSlider.value),
            max_tokens: 1024,
        });

        let fullReply = "";
        for await (const chunk of chunks) {
            const delta = chunk.choices[0]?.delta?.content || "";
            fullReply += delta;
            botBubble.textContent = fullReply;
            dom.chatBox.scrollTop = dom.chatBox.scrollHeight;
        }

        // Generate Citations Badge
        if (relevantContext) {
            const uniqueSources = [...new Set(topK.map(k => k.source))];
            const sourcesHtml = uniqueSources.map(s => `<span class="citation">source: ${s}</span>`).join('');
            const citationDiv = document.createElement('div');
            citationDiv.innerHTML = sourcesHtml;
            // Append badges to the LAST bot message (which is the one we just filled)
            dom.chatBox.lastElementChild.appendChild(citationDiv);
            dom.chatBox.scrollTop = dom.chatBox.scrollHeight;
        }

        // Update history (We store the ORIGINAL user text to keep history clean/readable, 
        // or the RAG version? Storing original is better for UI, but model needs context.
        // Let's store original text but the model saw the hidden context.)
        // Actually, simple way: Store what we sent.
        window.conversationHistory.push({ role: "user", content: finalUserMessage });
        window.conversationHistory.push({ role: "assistant", content: fullReply });

        // Bonus: TTS
        if (dom.ttsToggle.checked) {
            speakText(fullReply);
        }

        enableInput(true);

    } catch (err) {
        console.error("Chat error:", err);
        addSystemMessage("Error generating response: " + err.message);
        enableInput(true);
    }
}

function cosineSimilarity(vecA, vecB) {
    if (!vecA || !vecB) return 0;
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dot += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/****************************************************************************
 * UI HELPERS & EVENTS
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
    // Don't focus if disabled, only if re-enabled
    if (enabled) setTimeout(() => dom.userInput.focus(), 100);
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

function speakText(text) {
    if (!text) return;
    if (synth.speaking) {
        synth.cancel();
    }
    const utterThis = new SpeechSynthesisUtterance(text);
    // Simple voice selection
    const voices = synth.getVoices();
    const enVoice = voices.find(v => v.name.includes("Google US English")) || voices.find(v => v.lang.startsWith("en"));
    if (enVoice) utterThis.voice = enVoice;
    synth.speak(utterThis);
}

function escapeHtml(text) {
    if (!text) return "";
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

/* Event Listeners */
dom.sendBtn.addEventListener('click', handleUserMessage);
dom.userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleUserMessage();
});
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

// Settings Controls
dom.tempSlider.addEventListener('input', (e) => {
    dom.tempValue.textContent = e.target.value;
});
dom.micBtn.addEventListener('click', toggleRecording);

// Start
window.addEventListener('DOMContentLoaded', () => {
    initializeEngine();
});
