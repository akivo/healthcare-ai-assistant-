// script.js — Healthcare AI Assistant Frontend Logic
// =====================================================
// This file handles all user interactions:
// 1. Sending questions to the API
// 2. Displaying responses with sources
// 3. Checking system health
// 4. Triggering document ingestion

const API_BASE = window.location.origin; // Points to our FastAPI server

// DOM element references
const chatArea = document.getElementById('chatArea');
const messagesEl = document.getElementById('messages');
const welcomeScreen = document.getElementById('welcomeScreen');
const questionInput = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');
const ingestBtn = document.getElementById('ingestBtn');
const ingestStatus = document.getElementById('ingestStatus');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const statusDetails = document.getElementById('statusDetails');

// ============================================================
// HEALTH CHECK — Called on page load
// Checks if API is running, Ollama is connected, docs are loaded
// ============================================================

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();

    const llmOk = data.llm?.status === 'connected';
    const docsLoaded = data.vector_store?.chunks_stored > 0;

    if (llmOk) {
      statusDot.className = 'status-dot healthy';
      statusText.textContent = 'System Online';
    } else {
      statusDot.className = 'status-dot warning';
      statusText.textContent = 'LLM Offline';
    }

    statusDetails.innerHTML = `
      <span>🤖 LLM: ${data.llm?.status || 'unknown'}</span>
      <span>📚 Chunks: ${data.vector_store?.chunks_stored || 0}</span>
      <span>🧠 Model: ${data.config?.llm_model || '-'}</span>
    `;

    if (!docsLoaded) {
      ingestStatus.innerHTML = '⚠️ No documents loaded. Click "Ingest Documents" first.';
      ingestStatus.style.color = '#f59e0b';
    }

  } catch (err) {
    statusDot.className = 'status-dot error';
    statusText.textContent = 'API Offline';
    statusDetails.innerHTML = '<span>Make sure the server is running</span>';
  }
}

// ============================================================
// INGEST DOCUMENTS — Calls POST /ingest
// ============================================================

ingestBtn.addEventListener('click', async () => {
  ingestBtn.disabled = true;
  ingestStatus.textContent = '⏳ Ingesting documents...';
  ingestStatus.style.color = '#94a3b8';

  try {
    const res = await fetch(`${API_BASE}/ingest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data_dir: './data' }),
    });

    const data = await res.json();

    if (res.ok) {
      ingestStatus.innerHTML = `✅ ${data.documents_processed} docs, ${data.chunks_created} chunks loaded`;
      ingestStatus.style.color = '#10b981';
      await checkHealth(); // Refresh health status
    } else {
      ingestStatus.textContent = `❌ Error: ${data.detail}`;
      ingestStatus.style.color = '#ef4444';
    }
  } catch (err) {
    ingestStatus.textContent = '❌ Failed to connect to API';
    ingestStatus.style.color = '#ef4444';
  } finally {
    ingestBtn.disabled = false;
  }
});

// ============================================================
// SEND QUESTION — Calls POST /ask
// ============================================================

async function sendQuestion() {
  const question = questionInput.value.trim();
  if (!question) return;

  // Hide welcome screen
  welcomeScreen.style.display = 'none';

  // Show user message
  appendMessage('user', question);

  // Clear input
  questionInput.value = '';
  questionInput.style.height = 'auto';
  sendBtn.disabled = true;

  // Show typing indicator
  const typingId = showTyping();

  try {
    const res = await fetch(`${API_BASE}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });

    const data = await res.json();
    removeTyping(typingId);

    if (res.ok) {
      appendAssistantMessage(data);
    } else {
      appendMessage('assistant', `❌ Error: ${data.detail || 'Something went wrong.'}`);
    }
  } catch (err) {
    removeTyping(typingId);
    appendMessage('assistant', '❌ Could not connect to the API. Make sure the server is running.');
  } finally {
    sendBtn.disabled = false;
    questionInput.focus();
  }
}

// ============================================================
// APPEND USER MESSAGE
// ============================================================

function appendMessage(role, text) {
  const div = document.createElement('div');
  div.className = `message ${role}`;

  const avatarLetter = role === 'user' ? 'U' : '🏥';
  div.innerHTML = `
    <div class="message-avatar">${avatarLetter}</div>
    <div class="message-body">
      <div class="message-bubble">${escapeHtml(text)}</div>
    </div>
  `;

  messagesEl.appendChild(div);
  scrollToBottom();
}

// ============================================================
// APPEND ASSISTANT MESSAGE (with sources + badges)
// ============================================================

function appendAssistantMessage(data) {
  const div = document.createElement('div');
  div.className = 'message assistant';

  const confidence = data.confidence || 'low';
  const route = data.route === 'appointment_tool' ? '📅 Appointment Tool' : '🔍 RAG Pipeline';
  const sourceId = `sources-${Date.now()}`;

  // Build sources HTML
  let sourcesHtml = '';
  if (data.sources && data.sources.length > 0) {
    const sourceItems = data.sources.map(src => `
      <div class="source-item">
        <div class="source-filename">📄 ${src.document}</div>
        <div class="source-chunk">"${src.chunk}"</div>
        <div class="source-score">Relevance: ${(src.relevance_score * 100).toFixed(0)}%</div>
      </div>
    `).join('');

    sourcesHtml = `
      <div class="sources" id="${sourceId}">
        <button class="sources-toggle" onclick="toggleSources('${sourceId}')">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="9 18 15 12 9 6"/></svg>
          ${data.sources.length} source${data.sources.length > 1 ? 's' : ''} cited
        </button>
        <div class="sources-list">
          ${sourceItems}
        </div>
      </div>
    `;
  }

  div.innerHTML = `
    <div class="message-avatar">🏥</div>
    <div class="message-body">
      <div class="message-bubble">${formatAnswer(data.answer)}</div>
      <div style="display:flex;align-items:center;flex-wrap:wrap;gap:4px;margin-top:6px;">
        <span class="confidence-badge confidence-${confidence}">
          ${confidence === 'high' ? '●' : confidence === 'medium' ? '◐' : '○'} ${confidence} confidence
        </span>
        <span class="route-badge">${route}</span>
      </div>
      ${sourcesHtml}
    </div>
  `;

  messagesEl.appendChild(div);
  scrollToBottom();
}

// ============================================================
// TOGGLE SOURCES — Show/hide source citations
// ============================================================

function toggleSources(containerId) {
  const container = document.getElementById(containerId);
  const toggle = container.querySelector('.sources-toggle');
  const list = container.querySelector('.sources-list');

  toggle.classList.toggle('open');
  list.classList.toggle('visible');
}

// ============================================================
// TYPING INDICATOR
// ============================================================

function showTyping() {
  const id = `typing-${Date.now()}`;
  const div = document.createElement('div');
  div.className = 'message assistant';
  div.id = id;
  div.innerHTML = `
    <div class="message-avatar">🏥</div>
    <div class="message-body">
      <div class="message-bubble" style="padding:14px 18px;">
        <div class="typing-indicator">
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
        </div>
      </div>
    </div>
  `;
  messagesEl.appendChild(div);
  scrollToBottom();
  return id;
}

function removeTyping(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

function scrollToBottom() {
  chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: 'smooth' });
}

function escapeHtml(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\n/g, '<br>');
}

function formatAnswer(text) {
  // Convert newlines and basic formatting
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
}

// Use a sample question from sidebar or chips
function useSample(el) {
  const text = el.textContent.replace(/^[^\w]+/, '').trim(); // Remove emoji prefix
  questionInput.value = text;
  questionInput.dispatchEvent(new Event('input'));
  questionInput.focus();
  // On mobile, close sidebar
  document.getElementById('sidebar').classList.remove('mobile-open');
}

// ============================================================
// EVENT LISTENERS
// ============================================================

// Send on button click
sendBtn.addEventListener('click', sendQuestion);

// Send on Enter (Shift+Enter for new line)
questionInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendQuestion();
  }
});

// Auto-resize textarea
questionInput.addEventListener('input', () => {
  questionInput.style.height = 'auto';
  questionInput.style.height = Math.min(questionInput.scrollHeight, 120) + 'px';
});

// Sidebar toggle
document.getElementById('sidebarToggle').addEventListener('click', () => {
  document.getElementById('sidebar').classList.toggle('collapsed');
});

// Mobile menu toggle
document.getElementById('mobileMenuBtn').addEventListener('click', () => {
  document.getElementById('sidebar').classList.toggle('mobile-open');
});

// ============================================================
// INIT — Run on page load
// ============================================================
checkHealth();
