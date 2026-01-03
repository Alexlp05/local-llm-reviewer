import * as webllm from "https://esm.run/@mlc-ai/web-llm";

/****************************************************************************
 * CONFIGURATION
 ****************************************************************************/
// We use Llama-3-8B-Instruct (quantized) or Phi-3 as requested.
// Using a specific commit/quantization from MLC-AI's prebuilt weights.
const SELECTED_MODEL = "Llama-3-8B-Instruct-q4f32_1-MLC"; 

// DOM Elements
const dom = {
    chatBox: document.getElementById('chat-box'),
    userInput: document.getElementById('user-input'),
    sendBtn: document.getElementById('send-btn'),
    statusBar: document.getElementById('status-bar'),
    statusText: document.getElementById('status-text'),
    progressBarContainer: document.querySelector('.progress-bar-container'),
    progressBar: document.getElementById('progress-bar')
};

let engine = null;

/****************************************************************************
 * INITIALIZATION
 ****************************************************************************/
async function initializeEngine() {
    updateStatus("Initializing WebLLM...", 0);

    const initProgressCallback = (report) => {
        // report has format: { progress: number, text: string }
        // or just plain text messages in some versions, but recent WebLLM uses object
        // We'll handle both just in case, but standard is `report.text`
        const label = report.text || report;
        // console.log("Init:", label);
        updateStatus(label, report.progress || 0);
    };

    try {
        updateStatus("Loading Model (this may take a while)...", 0.01);
        
        // Create the engine
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
 * CHAT LOGIC
 ****************************************************************************/
async function handleUserMessage() {
    const text = dom.userInput.value.trim();
    if (!text || !engine) return;

    // 1. Add User Message to UI
    addUserMessage(text);
    dom.userInput.value = "";
    enableInput(false);

    try {
        // 2. Prepare conversation (we just send the session history implicit or explicit)
        // WebLLM engine maintains state if we use the chat completion API correctly or persist messages.
        // For simple chat, we can just push the new message to a local history array if we wanted manual control,
        // but engine.chat.completions.create is stateless unless we pass history. 
        // HOWEVER, `CreateMLCEngine` returns an engine that can hold state if used with `chat.completions`? 
        // Actually, WebLLM 'engine' is usually stateless request-response unless we manage history.
        // Wait, `MLCEngine` acts like OpenAI API. We need to maintain history array.
        
        if (!window.conversationHistory) {
            window.conversationHistory = [
                { role: "system", content: "You are a helpful academic assistant. Answer questions concisely." }
            ];
        }

        window.conversationHistory.push({ role: "user", content: text });

        // 3. Create a placeholder for Bot Message
        const botMessageId = addBotMessage(""); 
        const botBubble = document.getElementById(botMessageId).querySelector('.bubble');
        
        // 4. Stream response
        const messages = window.conversationHistory;
        
        const chunks = await engine.chat.completions.create({
            messages,
            stream: true,
            max_tokens: 512, // Limit response length for speed
        });

        let fullReply = "";
        
        for await (const chunk of chunks) {
            const delta = chunk.choices[0]?.delta?.content || "";
            fullReply += delta;
            botBubble.textContent = fullReply;
            dom.chatBox.scrollTop = dom.chatBox.scrollHeight;
        }

        // 5. Update history
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
    
    // Progress is 0.0 to 1.0 (sometimes string check)
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
    return text.replace(/&/g, "&amp;")
               .replace(/</g, "&lt;")
               .replace(/>/g, "&gt;")
               .replace(/"/g, "&quot;")
               .replace(/'/g, "&#039;");
}

/****************************************************************************
 * EVENT LISTENERS
 ****************************************************************************/
dom.sendBtn.addEventListener('click', handleUserMessage);
dom.userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleUserMessage();
});

// Start initialization on load
window.addEventListener('DOMContentLoaded', () => {
    initializeEngine();
});
