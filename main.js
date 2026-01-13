import * as webllm from "https://esm.run/@mlc-ai/web-llm";
import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

/****************************************************************************
 * CONFIGURATION
 ****************************************************************************/

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
    micBtn: document.getElementById('mic-btn'),
    memFileCount: document.getElementById('total-docs'),
    memChunkCount: document.getElementById('total-chunks'),
    fileList: document.getElementById('file-list'),
    contextSelect: document.getElementById('context-select'), // NEW
    personaSelect: document.getElementById('persona-select'), // NEW
    modelSelector: document.getElementById('model-selector')
};

let engine = null;
let extractor = null; // Transformers.js pipeline
let transcriber = null; // Whisper pipeline
let vectorStore = []; // To store chunks { id, text, vector, source }
let loadedFiles = new Set(); // Track loaded filenames
let activeDocument = null; // Filter for RAG
let chatSessions = { "global": [] }; // { sessionId: [ {role, content, type} ] }
let activeSessionId = "global";

// Initialize conversation history if not exists (though we rely on chatSessions source of truth mainly)
// We will reconstruct the messages array for the LLM from chatSessions each turn.

const synth = window.speechSynthesis;
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];

/****************************************************************************
 * INITIALIZATION
 ****************************************************************************/


async function initializeEngine() {
    const modelId = dom.modelSelector.value;
    updateStatus(`Initializing ${modelId}...`, 0);

    const initProgressCallback = (report) => {
        const label = report.text || report;
        updateStatus(label, report.progress || 0);
    };

    try {
        // Load WebLLM
        updateStatus(`Loading Chat Model (${modelId})...`, 0.01);
        engine = await webllm.CreateMLCEngine(
            modelId,
            { initProgressCallback: initProgressCallback }
        );

        // Load Transformers.js Embedding Model (only once)
        if (!extractor) {
            updateStatus("Loading Embedding Model (Transformers.js)...", 0.5);
            extractor = await pipeline('feature-extraction', EMBEDDING_MODEL);
        }

        updateStatus(`Ready to chat! (${modelId})`, 1);
        enableInput(true);
        addSystemMessage(`Model loaded: ${modelId}`);

    } catch (err) {
        console.error("Initialization error:", err);
        updateStatus("Error loading models: " + err.message, 0);
        addSystemMessage("Error: " + err.message);
    }
}

async function reloadEngine() {
    if (engine) {
        // webllm engine doesn't have an explicit 'unload' that we strictly need to call if we just overwrite, 
        // but let's be safe and just re-initialize.
        // engine.unload(); // If supported in future
        engine = null;
    }
    await initializeEngine();
}

/****************************************************************************
 * PDF HANDLING & CHUNKING
 ****************************************************************************/
async function handleFile(file) {
    if (file.type !== 'application/pdf') {
        alert("Please upload a PDF file.");
        return;
    }

    if (loadedFiles.has(file.name)) {
        addSystemMessage(`File "${file.name}" is already loaded.`);
        return;
    }

    addSystemMessage(`Reading ${file.name}...`);
    updateStatus("Reading PDF...", 0.1);

    try {
        const text = await extractTextFromPDF(file);
        addSystemMessage(`PDF Loaded. Length: ${text.length} characters.`);

        // Chunking
        const chunks = chunkText(text, 500, 50); // 500 chars, 50 overlap

        // Create temporary store for this file
        const newChunks = chunks.map((chunk, index) => ({
            id: `${file.name}-${index}`,
            text: chunk,
            vector: null,
            source: file.name
        }));

        updateStatus("Generating Embeddings...", 0.2);

        // Generate Embeddings for NEW chunks
        await embedChunks(newChunks);

        // Add to Global Store
        vectorStore.push(...newChunks);
        loadedFiles.add(file.name);

        // Update UI
        updateMemoryUI();

        dom.docStatus.textContent = `Added: ${file.name} (+${newChunks.length} chunks)`;
        addSystemMessage(`Embeddings generated for ${newChunks.length} chunks. Added to Memory Bank.`);
        updateStatus("Ready to chat!", 1);

    } catch (err) {
        console.error("PDF Error:", err);
        addSystemMessage("Error reading PDF: " + err.message);
        updateStatus("Error reading PDF", 0);
    }
}

async function embedChunks(chunksToEmbed) {
    if (!extractor) {
        addSystemMessage("Embedding model not loaded yet.");
        return;
    }

    const total = chunksToEmbed.length;
    for (let i = 0; i < total; i++) {
        const chunk = chunksToEmbed[i];
        const output = await extractor(chunk.text, { pooling: 'mean', normalize: true });
        chunk.vector = Array.from(output.data);

        if (i % 10 === 0) {
            updateStatus(`Embedding Chunks (${i + 1}/${total})...`, (i + 1) / total);
        }
    }
}

function updateMemoryUI() {
    dom.memFileCount.textContent = loadedFiles.size;
    dom.memChunkCount.textContent = vectorStore.length;

    // 1. Refresh Dropdown (Primary Control)
    // Save current selection (if valid)
    const currentVal = activeDocument === null ? "global" : activeDocument;

    dom.contextSelect.innerHTML = '<option value="global">🌐 Global Context (All Files)</option>';

    loadedFiles.forEach(filename => {
        const option = document.createElement('option');
        option.value = filename;
        option.textContent = `📄 ${filename}`;
        dom.contextSelect.appendChild(option);
    });

    // Restore selection
    dom.contextSelect.value = currentVal;


    // 2. Rebuild File List (Informational View)
    dom.fileList.innerHTML = '';

    loadedFiles.forEach(filename => {
        const count = vectorStore.filter(c => c.source === filename).length;
        const li = document.createElement('li');
        // Visual highlight sync
        const isActive = activeDocument === filename;
        li.className = `file-item ${isActive ? 'active' : ''} cursor-default`; // reduce pointer if only info

        // Let's keep click-to-switch on list item too as a convenience? 
        // User requested "Dropdown", but standard behavior allows both. 
        // Let's Sync them. Click list -> Update Dropdown. Change Dropdown -> Update List highlight.

        li.innerHTML = `
            <span class="truncate">${escapeHtml(filename)}</span>
            <span class="badge text-[10px] bg-slate-700 px-1 rounded">${count} chunks</span>
        `;

        li.addEventListener('click', () => {
            // Sync Dropdown
            dom.contextSelect.value = filename;
            changeContext(filename);
        });

        dom.fileList.appendChild(li);
    });
}

function changeContext(value) {
    if (value === "global") {
        if (activeDocument !== null) {
            activeDocument = null;
            switchSession("global");
            addSystemMessage("Switched to Global Context.", false); // false = Ephemeral (don't save)
            updateMemoryUI(); // to highlight list
        }
    } else {
        if (activeDocument !== value) {
            activeDocument = value;
            if (!chatSessions[value]) chatSessions[value] = [];
            switchSession(value);
            addSystemMessage(`Switched context to: ${value}`, false); // false = Ephemeral (don't save)
            updateMemoryUI(); // to highlight list
        }
    }
}

function switchSession(sessionId) {
    activeSessionId = sessionId;

    // Clear UI
    dom.chatBox.innerHTML = '';

    // Re-render history
    const history = chatSessions[sessionId] || [];
    history.forEach(msg => {
        // FILTER: Don't show old "Switched..." messages
        if (msg.role === 'system' && (msg.content.startsWith("Switched") || msg.content.startsWith("Switching"))) {
            return;
        }

        if (msg.role === 'user') {
            addUserMessage(msg.content, false); // false = don't save again
        } else if (msg.role === 'assistant') {
            addBotMessage(msg.content, false); // false = don't save again
        } else if (msg.role === 'system') {
            addSystemMessage(msg.content, false);
        }
    });

    dom.chatBox.scrollTop = dom.chatBox.scrollHeight;
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

    addUserMessage(text, true); // true = save to session
    dom.userInput.value = "";
    enableInput(false);

    // CAPTURE SESSION ID: Ensure we save to the session where the request started
    const targetSessionId = activeSessionId;

    try {
        let systemPrompt = dom.systemPrompt.value;
        let relevantContext = "";
        let topK = [];

        // RAG STEP: If we have documents, find relevant chunks
        if (vectorStore.length > 0 && extractor) {
            // addSystemMessage("Searching documents..."); // Optional: too noisy for history?

            // 1. Embed Query
            const output = await extractor(text, { pooling: 'mean', normalize: true });
            const queryVector = Array.from(output.data);

            // RAG FILTER: Active Document
            const searchPool = activeDocument
                ? vectorStore.filter(chunk => chunk.source === activeDocument)
                : vectorStore;

            // 2. Cosine Similarity Search
            const similarities = searchPool.map(chunk => ({
                ...chunk,
                score: cosineSimilarity(queryVector, chunk.vector)
            }));

            // 3. Sort & Top-K
            similarities.sort((a, b) => b.score - a.score);
            topK = similarities.slice(0, 10); // Top 10 chunks (Better context for 6 papers)

            // 4. Construct Context
            relevantContext = topK.map(chunk => `[Context]: ${chunk.text}`).join("\n\n");
            console.log("RAG Context:", topK);

            systemPrompt += "\n\nCONTEXT:\n" + relevantContext;
        }

        // WebLLM logic

        // 1. Construct System Message
        // We do strictly one system message at the start.
        // If we have RAG context, we can append it to the Request or the System Prompt.
        // Best practice for "Chat with Doc": System Prompt + "Context: ... "

        let finalSystemPrompt = systemPrompt; // Value from DOM already includes RAG Context appended in previous lines (lines 391) -> wait, logic check below.

        // Logic correction: Lines 360-392 modify a local variable 'systemPrompt'. Perfect.

        // 2. Construct Conversation History
        // We take the current session's history EXCLUDING the latest user message we just added? 
        // Actually, 'chatSessions[activeSessionId]' already has the user message? 
        // Let's check: addUserMessage(text, true) -> pushed to chatSessions.

        // So we need to map chatSessions to WebLLM format.
        // filter out 'system' messages from history because we are constructing a fresh MAIN system message.
        // (Unless we want to keep previous system messages? Usually no, we want the LATEST config).

        const currentSessionMessages = chatSessions[activeSessionId] || [];

        const historyForModel = currentSessionMessages
            .filter(msg => msg.role !== 'system') // We provide a fresh system prompt each turn
            .map(msg => ({ role: msg.role, content: msg.content }));

        // The latest user message is ALREADY in historyForModel because we called addUserMessage(text, true) at line 355.
        // Wait, if we use historyForModel, does it include the context?
        // We decided in the plan: "Inject context into the User message" OR "System Prompt".
        // Line 410 says: "finalUserMessage = ... relevantContext ...". 
        // But logic at 355 saved the RAW text.

        // OPTION A: Modifying the last user message in the array to include context (Invisible to UI, visible to LLM).
        // OPTION B: Putting it in System Prompt.

        // Let's go with OPTION A (Context in User Message) for stronger attention, OR System Prompt. 
        // The previous code 391 appended to systemPrompt. Let's stick to that for "Lit Reviewer" style.
        // "System: You are a researcher... Context: ... "

        const messages = [
            { role: "system", content: finalSystemPrompt },
            ...historyForModel
        ];

        // NOTE: If the user just sent a message, it is the LAST item in historyForModel.
        // If we want to attach context specifically to that message, we could. 
        // But effectively, System Prompt context works well for "Answer based on this data".

        // Stream response
        const botMessageId = addBotMessage("");
        const botContentDiv = document.getElementById(botMessageId).querySelector('.message-content');

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
            // Markdown Render on the fly (might be heavy but okay for decent machines)
            botContentDiv.innerHTML = marked.parse(fullReply);
            dom.chatBox.scrollTop = dom.chatBox.scrollHeight;
        }

        // Generate Citations Badge
        if (relevantContext) {
            const uniqueSources = [...new Set(topK.map(k => k.source))];
            const sourcesHtml = uniqueSources.map(s => `<span class="citation-badge">source: ${s}</span>`).join('');
            const citationDiv = document.createElement('div');
            citationDiv.innerHTML = sourcesHtml;
            // Append badges to contents
            botContentDiv.appendChild(citationDiv);
            dom.chatBox.scrollTop = dom.chatBox.scrollHeight;
        }

        // Save Bot Message to Session (Targeted)
        // We use targetSessionId to ensure it goes to the correct history even if user switched away
        if (!chatSessions[targetSessionId]) chatSessions[targetSessionId] = [];
        chatSessions[targetSessionId].push({ role: "assistant", content: fullReply });

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

function addSystemMessage(text, save = true) {
    const div = document.createElement('div');
    div.className = 'message-row system bg-slate-800/20 border-b border-slate-800';
    div.innerHTML = `
        <div class="avatar system">⚙️</div>
        <div class="message-content text-sm text-slate-400 pt-1">
            ${escapeHtml(text)}
        </div>
    `;
    dom.chatBox.appendChild(div);
    dom.chatBox.scrollTop = dom.chatBox.scrollHeight;

    if (save) {
        if (!chatSessions[activeSessionId]) chatSessions[activeSessionId] = [];
        chatSessions[activeSessionId].push({ role: "system", content: text });
    }
}

function addUserMessage(text, save = true) {
    const div = document.createElement('div');
    div.className = 'message-row user';
    // User message is plain text usually, or we can use marked too if we want user markdown support. 
    // Let's stick to escapeHtml for user input to be safe/simple, or marked if they paste code.
    // Let's use simple text for user to mimic "Input".
    div.innerHTML = `
        <div class="avatar user">👤</div>
        <div class="message-content">
            ${escapeHtml(text)}
        </div>
    `;
    dom.chatBox.appendChild(div);
    dom.chatBox.scrollTop = dom.chatBox.scrollHeight;

    if (save) {
        if (!chatSessions[activeSessionId]) chatSessions[activeSessionId] = [];
        chatSessions[activeSessionId].push({ role: "user", content: text });
    }
}

function addBotMessage(initialText, save = true) {
    const id = "bot-msg-" + Date.now();
    const div = document.createElement('div');
    div.id = id;
    div.className = 'message-row bot';
    // Bot message content will be updated via streaming
    div.innerHTML = `
        <div class="avatar bot">🤖</div>
        <div class="message-content">
            ${marked.parse(initialText)}
        </div>
    `;
    dom.chatBox.appendChild(div);
    dom.chatBox.scrollTop = dom.chatBox.scrollHeight;

    // Note: Bot messages are streamed, so we usually save them AFTER completion in handleUserMessage
    // But if we are just re-rendering (save=false), we present it immediately.
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
dom.modelSelector.addEventListener('change', reloadEngine);
dom.contextSelect.addEventListener('change', (e) => {
    changeContext(e.target.value);
});

// --- PERSONA SELECTOR ---
const PERSONAS = {
    scholar: `You are an expert academic researcher.
When asked to review literature:
1. SYNTHESIZE themes across papers (e.g. "Paper A argues X, while Paper B suggests Y").
2. Use a structured format: Introduction, Comparison of Approaches, Conclusion.
3. Be specific and cite your sources.`,

    simplifier: `You are a skilled teacher who explains complex concepts simply (ELI5).
1. Use analogies and everyday language.
2. Avoid jargon or explain it if necessary.
3. Use bullet points for clarity.`,

    critic: `You are a critical academic reviewer.
1. Focus on identifying methodology flaws, limitations, and contradictions.
2. Challenge the authors' assumptions.
3. Point out what is missing or under-explored in the research.`
};

if (dom.personaSelect) {
    dom.personaSelect.addEventListener('change', (e) => {
        console.log("Persona changed to:", e.target.value);
        const p = PERSONAS[e.target.value];
        if (p) {
            dom.systemPrompt.value = p;
        }
    });
} else {
    console.error("Persona selector not found in DOM");
}

// Initialize (Auto-load Scholar)
// Optional: if (dom.systemPrompt.value === "") dom.systemPrompt.value = PERSONAS.scholar;

// Start
window.addEventListener('DOMContentLoaded', () => {
    initializeEngine();
});
