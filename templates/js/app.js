// ===== State =====
let knowledgeBases = [];
let conversations = [];
let currentConversation = null;
let currentKbName = '';
let selectedKbs = new Set();
let uploadFilesList = [];
let eventSource = null;

// ===== Init =====
document.addEventListener('DOMContentLoaded', () => {
    loadKnowledgeBases();
    loadConversations();
    updateStats();
    setupAutoResize();
    setupDragDrop();
});

// ===== Navigation =====
function switchPage(page) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.getElementById('page-' + page).classList.add('active');
    document.querySelector('[data-page="' + page + '"]').classList.add('active');

    const titles = {
        dashboard: '总览',
        knowledge: '知识库管理',
        chat: '智能问答',
        history: '对话历史'
    };
    document.getElementById('page-title').textContent = titles[page] || '智识库';

    if (page === 'knowledge') loadKnowledgeBases();
    if (page === 'chat') {
        loadKnowledgeBases();
        renderKbSelector();
    }
    if (page === 'history') loadConversations();
}

// ===== API Helpers =====
async function apiGet(url) {
    const res = await fetch(url);
    return res.json();
}

async function apiPost(url, data) {
    const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    return res.json();
}

// ===== Knowledge Bases =====
async function loadKnowledgeBases() {
    try {
        const res = await apiGet('/api/kb/list');
        if (res.success) {
            knowledgeBases = res.data;
            document.getElementById('kb-count').textContent = knowledgeBases.length;
            renderKbList();
            renderDashboardKbList();
            updateStats();
        }
    } catch (e) {
        showToast('error', '加载失败', '无法获取知识库列表');
    }
}

function renderKbList() {
    const container = document.getElementById('kb-list');
    if (knowledgeBases.length === 0) {
        container.innerHTML = `
            <div class="empty-state" style="grid-column: 1 / -1;">
                <div class="empty-icon">📚</div>
                <div class="empty-title">暂无知识库</div>
                <div class="empty-desc">点击右上角按钮创建你的第一个知识库</div>
            </div>
        `;
        return;
    }

    container.innerHTML = knowledgeBases.map(kb => `
        <div class="card" onclick="showKbDetail('${kb.name}')">
            <div class="card-header">
                <div class="card-icon">📁</div>
                <div class="card-menu" onclick="event.stopPropagation()">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="5" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="12" cy="19" r="1"/></svg>
                </div>
            </div>
            <div class="card-body">
                <div class="card-title">${kb.name}</div>
                <div class="card-desc">${kb.file_count} 个文件 · ${kb.is_built ? '已构建' : '未构建'}</div>
            </div>
            <div class="card-footer">
                <span class="card-meta">${kb.knowledge_path}</span>
                <span class="card-status ${kb.is_built ? 'status-built' : 'status-pending'}">
                    ${kb.is_built ? '✓ 已就绪' : '○ 待构建'}
                </span>
            </div>
        </div>
    `).join('');
}

function renderDashboardKbList() {
    const container = document.getElementById('dashboard-kb-list');
    const recent = knowledgeBases.slice(0, 4);
    if (recent.length === 0) {
        container.innerHTML = `
            <div class="empty-state" style="grid-column: 1 / -1;">
                <div class="empty-icon">📚</div>
                <div class="empty-title">暂无知识库</div>
                <div class="empty-desc">创建知识库开始使用</div>
            </div>
        `;
        return;
    }
    container.innerHTML = recent.map(kb => `
        <div class="card" onclick="showKbDetail('${kb.name}')">
            <div class="card-header">
                <div class="card-icon">📁</div>
                <div class="card-menu" onclick="event.stopPropagation()">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="5" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="12" cy="19" r="1"/></svg>
                </div>
            </div>
            <div class="card-body">
                <div class="card-title">${kb.name}</div>
                <div class="card-desc">${kb.file_count} 个文件 · ${kb.is_built ? '已构建' : '未构建'}</div>
            </div>
            <div class="card-footer">
                <span class="card-meta">${kb.knowledge_path}</span>
                <span class="card-status ${kb.is_built ? 'status-built' : 'status-pending'}">
                    ${kb.is_built ? '✓ 已就绪' : '○ 待构建'}
                </span>
            </div>
        </div>
    `).join('');
}

function updateStats() {
    document.getElementById('stat-kb-count').textContent = knowledgeBases.length;
    document.getElementById('stat-file-count').textContent = knowledgeBases.reduce((a, b) => a + b.file_count, 0);
    document.getElementById('stat-built-count').textContent = knowledgeBases.filter(k => k.is_built).length;
}

// ===== Modals =====
function showModal(id) {
    document.getElementById(id).classList.add('active');
}

function closeModal(id) {
    document.getElementById(id).classList.remove('active');
}

function showCreateKbModal() {
    document.getElementById('new-kb-name').value = '';
    showModal('create-kb-modal');
}

async function createKnowledgeBase() {
    const name = document.getElementById('new-kb-name').value.trim();
    if (!name) {
        showToast('warning', '请输入名称', '知识库名称不能为空');
        return;
    }
    try {
        const res = await apiPost('/api/kb/create', { name });
        if (res.success) {
            showToast('success', '创建成功', `知识库「${name}」已创建`);
            closeModal('create-kb-modal');
            loadKnowledgeBases();
        } else {
            showToast('error', '创建失败', res.message);
        }
    } catch (e) {
        showToast('error', '创建失败', '网络错误');
    }
}

function showUploadModal(kbName) {
    currentKbName = kbName;
    document.getElementById('upload-kb-name').textContent = kbName;
    uploadFilesList = [];
    renderUploadFileList();
    closeModal('kb-detail-modal');
    showModal('upload-modal');
}

async function showKbDetail(kbName) {
    currentKbName = kbName;
    document.getElementById('detail-kb-name').textContent = kbName;
    document.getElementById('_build-progress').style.display = 'none';

    try {
        const res = await apiGet('/api/kb/files?name=' + encodeURIComponent(kbName));
        if (res.success) {
            const list = document.getElementById('kb-file-list');
            if (res.data.length === 0) {
                list.innerHTML = '<div class="empty-state" style="padding: 20px;"><div class="empty-desc">暂无文件</div></div>';
            } else {
                list.innerHTML = res.data.map(f => `
                    <div class="file-item">
                        <div class="file-icon">📄</div>
                        <div class="file-info">
                            <div class="file-name">${f.name}</div>
                            <div class="file-size">${formatBytes(f.size)} · ${new Date(f.modified * 1000).toLocaleString()}</div>
                        </div>
                    </div>
                `).join('');
            }
        }
    } catch (e) {
        showToast('error', '加载失败', '无法获取文件列表');
    }
    showModal('kb-detail-modal');
}

// ===== File Upload =====
function setupDragDrop() {
    const zone = document.getElementById('upload-zone');
    zone.addEventListener('dragover', e => {
        e.preventDefault();
        zone.classList.add('dragover');
    });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', e => {
        e.preventDefault();
        zone.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });
}

function handleFileSelect(e) {
    handleFiles(e.target.files);
}

function handleFiles(files) {
    uploadFilesList = Array.from(files);
    renderUploadFileList();
}

function renderUploadFileList() {
    const container = document.getElementById('upload-file-list');
    if (uploadFilesList.length === 0) {
        container.innerHTML = '';
        return;
    }
    container.innerHTML = uploadFilesList.map((f, i) => `
        <div class="file-item">
            <div class="file-icon">📄</div>
            <div class="file-info">
                <div class="file-name">${f.name}</div>
                <div class="file-size">${formatBytes(f.size)}</div>
            </div>
            <div class="file-actions">
                <button class="btn btn-ghost btn-icon btn-sm" onclick="removeUploadFile(${i})">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                </button>
            </div>
        </div>
    `).join('');
}

function removeUploadFile(index) {
    uploadFilesList.splice(index, 1);
    renderUploadFileList();
}

async function uploadFiles() {
    if (uploadFilesList.length === 0) {
        showToast('warning', '未选择文件', '请先选择要上传的文件');
        return;
    }

    for (const file of uploadFilesList) {
        const formData = new FormData();
        formData.append('kb_name', currentKbName);
        formData.append('file', file);

        try {
            const res = await fetch('/api/kb/upload', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (data.success) {
                showToast('success', '上传成功', `${file.name} 已上传`);
            } else {
                showToast('error', '上传失败', data.message);
            }
        } catch (e) {
            showToast('error', '上传失败', '网络错误');
        }
    }

    uploadFilesList = [];
    renderUploadFileList();
    loadKnowledgeBases();
    showKbDetail(currentKbName);
}

// ===== Build Knowledge Base =====
function buildKnowledgeBase(kbName) {
    if (eventSource) eventSource.close();

    document.getElementById('build-progress').style.display = 'block';
    document.getElementById('progress-fill').style.width = '0%';
    document.getElementById('progress-text').textContent = '准备构建...';

    eventSource = new EventSource('/build_progress/' + encodeURIComponent(kbName));

    eventSource.onmessage = e => {
        const data = JSON.parse(e.data);
        document.getElementById('progress-fill').style.width = data.progress + '%';
        document.getElementById('progress-text').textContent = data.message;

        if (data.progress === 100) {
            eventSource.close();
            if (data.success) {
                showToast('success', '构建完成', data.message);
                loadKnowledgeBases();
            } else {
                showToast('error', '构建失败', data.message);
            }
        }
    };

    eventSource.onerror = () => {
        eventSource.close();
        showToast('error', '连接错误', '构建进度连接中断');
    };
}

// ===== Delete Knowledge Base =====
async function deleteKnowledgeBase(kbName) {
    if (!confirm(`确定要删除知识库「${kbName}」吗？此操作不可恢复。`)) return;

    try {
        const res = await apiPost('/api/kb/delete', { name: kbName });
        if (res.success) {
            showToast('success', '删除成功', `知识库「${kbName}」已删除`);
            closeModal('kb-detail-modal');
            loadKnowledgeBases();
        } else {
            showToast('error', '删除失败', res.message);
        }
    } catch (e) {
        showToast('error', '删除失败', '网络错误');
    }
}

// ===== Chat =====
function renderKbSelector() {
    const container = document.getElementById('chat-kb-selector');
    if (knowledgeBases.length === 0) {
        container.innerHTML = '<span style="font-size: 12px; color: var(--text-tertiary);">暂无可用知识库</span>';
        return;
    }
    container.innerHTML = knowledgeBases.map(kb => `
        <div class="kb-chip ${selectedKbs.has(kb.name) ? 'selected' : ''}" onclick="toggleKb('${kb.name}')">
            <span class="check">✓</span>
            <span>${kb.name}</span>
        </div>
    `).join('');
}

function toggleKb(name) {
    if (selectedKbs.has(name)) {
        selectedKbs.delete(name);
    } else {
        selectedKbs.add(name);
    }
    renderKbSelector();
}

function setupAutoResize() {
    const textarea = document.getElementById('chat-input');
    textarea.addEventListener('input', () => {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    });
}

function handleChatKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

async function createNewConversation() {
    try {
        const res = await apiPost('/api/conversation', { title: '新对话' });
        if (res.success) {
            currentConversation = res.data;
            conversations.unshift(res.data);
            renderConversationList();
            clearChatMessages();
            showToast('success', '创建成功', '新对话已创建');
        }
    } catch (e) {
        showToast('error', '创建失败', '无法创建新对话');
    }
}

function clearChatMessages() {
    document.getElementById('chat-messages').innerHTML = `
        <div class="empty-state">
            <div class="empty-icon">💬</div>
            <div class="empty-title">开始新的对话</div>
            <div class="empty-desc">选择知识库，输入问题，获取智能回答</div>
        </div>
    `;
}

async function loadConversations() {
    try {
        const res = await apiGet('/api/history');
        if (res.success) {
            conversations = res.data;
            document.getElementById('stat-conv-count').textContent = conversations.length;
            renderConversationList();
            renderHistoryList();
        }
    } catch (e) {
        console.error('Failed to load conversations', e);
    }
}

function renderConversationList() {
    const container = document.getElementById('conversation-list');
    if (conversations.length === 0) {
        container.innerHTML = '<div class="empty-state" style="padding: 20px;"><div class="empty-desc">暂无对话</div></div>';
        return;
    }
    container.innerHTML = conversations.map(c => `
        <div class="chat-item ${currentConversation && currentConversation.id === c.id ? 'active' : ''}" onclick="loadConversation('${c.id}')">
            <div class="chat-item-title">${c.title}</div>
            <div class="chat-item-actions">
                <button class="btn btn-ghost btn-icon btn-sm chat-item-edit" onclick="event.stopPropagation(); startRenameConversation('${c.id}', '${c.title}')" title="重命名">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15 4 16h7"/></svg>
                </button>
            </div>
        </div>
    `).join('');
}

function startRenameConversation(id, currentTitle) {
    const newTitle = prompt('请输入新的对话名称:', currentTitle);
    if (newTitle && newTitle.trim() && newTitle !== currentTitle) {
        renameConversation(id, newTitle.trim());
    }
}

async function renameConversation(id, newTitle) {
    try {
        const res = await fetch('/api/conversation/' + id, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: newTitle })
        });
        const data = await res.json();
        if (data.success) {
            const conv = conversations.find(c => c.id === id);
            if (conv) conv.title = newTitle;
            renderConversationList();
            renderHistoryList();
            if (currentConversation && currentConversation.id === id) {
                currentConversation.title = newTitle;
            }
            showToast('success', '重命名成功', '对话名称已更新');
        } else {
            showToast('error', '重命名失败', data.message);
        }
    } catch (e) {
        showToast('error', '重命名失败', '网络错误');
    }
}

async function loadConversation(id) {
    try {
        const res = await apiGet('/api/conversation/' + id);
        if (res.success) {
            currentConversation = res.data;
            renderConversationList();
            renderMessages(res.data.messages);
        }
    } catch (e) {
        showToast('error', '加载失败', '无法加载对话');
    }
}

function renderMessages(messages) {
    const container = document.getElementById('chat-messages');
    if (!messages || messages.length === 0) {
        clearChatMessages();
        return;
    }
    container.innerHTML = messages.map(m => `
        <div class="message message-user">
            <div class="message-bubble">${escapeHtml(m.question)}</div>
            <div class="message-meta">${m.time}</div>
        </div>
        <div class="message message-assistant">
            <div class="message-bubble">
                ${m.thinking ? `<div class="thinking-box"><div class="thinking-label"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>思考过程</div>${escapeHtml(m.thinking)}</div>` : ''}
                <div class="markdown-content">${parseMarkdown(m.answer)}</div>
            </div>
            <div class="message-meta">AI 助手 · ${m.time}</div>
        </div>
    `).reverse().join('');
    container.scrollTop = container.scrollHeight;
}

let pendingQuestion = '';
let pendingEnableWeb = false;
let retrievedChunks = [];
let selectedChunks = new Set();

async function sendMessage() {
    const input = document.getElementById('chat-input');
    const question = input.value.trim();
    if (!question) return;

    if (selectedKbs.size === 0) {
        showToast('warning', '未选择知识库', '请至少选择一个知识库');
        return;
    }

    pendingQuestion = question;
    pendingEnableWeb = document.getElementById('enable-web').checked;
    input.value = '';
    input.style.height = 'auto';

    const container = document.getElementById('chat-messages');
    if (container.querySelector('.empty-state')) {
        container.innerHTML = '';
    }

    const userMsg = document.createElement('div');
    userMsg.className = 'message message-user';
    userMsg.innerHTML = `<div class="message-bubble">${escapeHtml(question)}</div><div class="message-meta">刚刚</div>`;
    container.appendChild(userMsg);
    container.scrollTop = container.scrollHeight;

    const searchingMsg = document.createElement('div');
    searchingMsg.className = 'message message-assistant';
    searchingMsg.innerHTML = `
        <div class="message-bubble">
            <div style="display: flex; align-items: center; gap: 8px; color: var(--text-secondary);">
                <div class="typing-indicator"><span></span><span></span><span></span></div>
                <span>正在检索知识库...</span>
            </div>
        </div>
    `;
    container.appendChild(searchingMsg);
    container.scrollTop = container.scrollHeight;

    try {
        const res = await apiPost('/api/retrieve', {
            question: question,
            selected_kbs: Array.from(selectedKbs)
        });

        container.removeChild(searchingMsg);

        if (res.error) {
            showToast('error', '检索失败', res.error);
            return;
        }

        retrievedChunks = res.knowledge_chunks || [];

        if (retrievedChunks.length === 0) {
            showToast('warning', '未找到相关知识', '将使用知识库全部内容生成回答');
            await generateAnswer([]);
        } else {
            showChunksModal(retrievedChunks, question);
        }

    } catch (e) {
        container.removeChild(searchingMsg);
        showToast('error', '发送失败', '网络错误');
    }
}

function showChunksModal(chunks, question) {
    document.getElementById('chunks-question').textContent = '问题: ' + question;
    selectedChunks = new Set();
    renderChunksList(chunks);
    document.getElementById('chunks-modal').classList.add('active');
}

function closeChunksModal() {
    document.getElementById('chunks-modal').classList.remove('active');
}

function renderChunksList(chunks) {
    const container = document.getElementById('chunks-list');

    if (chunks.length === 0) {
        container.innerHTML = `
            <div class="chunks-empty">
                <div class="chunks-empty-icon">📚</div>
                <div>未检索到相关知识切片</div>
            </div>
        `;
        return;
    }

    container.innerHTML = chunks.map((chunk, index) => {
        const isObject = typeof chunk === 'object';
        const source = isObject ? (chunk.source || '未知来源') : '知识库内容';
        const score = isObject ? (chunk.score || 0) : 0;
        const content = isObject ? (chunk.content || chunk) : chunk;
        const preview = content.length > 150 ? content.substring(0, 150) + '...' : content;

        return `
            <div class="chunk-item" data-index="${index}" onclick="toggleChunk(${index})">
                <div class="chunk-header-row">
                    <div class="chunk-checkbox">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>
                    </div>
                    <div class="chunk-info">
                        <span class="chunk-source">${escapeHtml(source)}</span>
                        <span class="chunk-score">相似度: ${(score * 100).toFixed(1)}%</span>
                    </div>
                </div>
                <div class="chunk-preview">${escapeHtml(preview)}</div>
            </div>
        `;
    }).join('');

    updateChunksCount();
}

function toggleChunk(index) {
    const items = document.querySelectorAll('.chunk-item');
    const item = items[index];

    if (selectedChunks.has(index)) {
        selectedChunks.delete(index);
        item.classList.remove('selected');
    } else {
        selectedChunks.add(index);
        item.classList.add('selected');
    }

    updateChunksCount();
    document.getElementById('select-all-chunks').checked = selectedChunks.size === retrievedChunks.length;
}

function toggleSelectAllChunks() {
    const selectAll = document.getElementById('select-all-chunks').checked;
    const items = document.querySelectorAll('.chunk-item');

    if (selectAll) {
        retrievedChunks.forEach((_, index) => {
            selectedChunks.add(index);
            items[index].classList.add('selected');
        });
    } else {
        selectedChunks.clear();
        items.forEach(item => item.classList.remove('selected'));
    }

    updateChunksCount();
}

function updateChunksCount() {
    document.getElementById('selected-chunks-count').textContent = selectedChunks.size;
}

async function confirmChunks() {
    const selected = Array.from(selectedChunks).map(index => retrievedChunks[index]);
    closeChunksModal();
    await generateAnswer(selected);
}

async function generateAnswer(knowledgeChunks) {
    const container = document.getElementById('chat-messages');

    const typingMsg = document.createElement('div');
    typingMsg.className = 'message message-assistant';
    typingMsg.innerHTML = `
        <div class="message-bubble">
            <div class="typing-indicator"><span></span><span></span><span></span></div>
        </div>
    `;
    container.appendChild(typingMsg);
    container.scrollTop = container.scrollHeight;

    try {
        const enableWeb = pendingEnableWeb;
        const selectedKnowledge = knowledgeChunks.filter(k => typeof k === 'object');

        const res = await apiPost('/api/chat', {
            question: pendingQuestion,
            selected_kbs: Array.from(selectedKbs),
            selected_knowledge: selectedKnowledge,
            enable_web: enableWeb
        });

        container.removeChild(typingMsg);

        if (res.error) {
            showToast('error', '请求失败', res.error);
            return;
        }

        const assistantMsg = document.createElement('div');
        assistantMsg.className = 'message message-assistant';

        let thinkingHtml = '';
        if (res.thinking) {
            thinkingHtml = `<div class="thinking-box"><div class="thinking-label"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>思考过程</div>${escapeHtml(res.thinking)}</div>`;
        }

        let sourcesHtml = '';
        if (res.local_knowledge && res.local_knowledge.length > 0) {
            const sources = res.local_knowledge.filter(k => typeof k === 'object');
            if (sources.length > 0) {
                sourcesHtml = `<div class="knowledge-tags">${sources.map(s => `<span class="knowledge-tag">${escapeHtml(s.source)}</span>`).join('')}</div>`;
            }
        }

        assistantMsg.innerHTML = `
            <div class="message-bubble">
                ${thinkingHtml}
                <div class="markdown-content">${parseMarkdown(res.final_answer)}</div>
                ${sourcesHtml}
            </div>
            <div class="message-meta">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align: middle; margin-right: 4px;"><path d="M12 2a10 10 0 1 0 10 10 4 4 0 0 1-5-5 4 4 0 0 1-5-5"/><path d="M8.5 8.5v.01"/><path d="M16 15.5v.01"/><path d="M12 12v.01"/><path d="M11 17v.01"/><path d="M7 14v.01"/></svg>
                AI 助手 · 刚刚
            </div>
        `;
        container.appendChild(assistantMsg);
        container.scrollTop = container.scrollHeight;

        if (currentConversation) {
            await apiPost('/api/conversation/' + currentConversation.id + '/message', {
                question: pendingQuestion,
                answer: res.final_answer,
                thinking: res.thinking || ''
            });
        }

    } catch (e) {
        container.removeChild(typingMsg);
        showToast('error', '生成失败', '网络错误');
    }
}

// ===== History =====
function renderHistoryList() {
    const container = document.getElementById('history-list');
    if (conversations.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">📜</div>
                <div class="empty-title">暂无历史记录</div>
                <div class="empty-desc">开始对话后将显示在这里</div>
            </div>
        `;
        return;
    }
    container.innerHTML = conversations.map(c => `
        <div class="card" style="margin-bottom: 12px; cursor: pointer;" onclick="switchPage('chat'); loadConversation('${c.id}')">
            <div class="card-body" style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <div class="card-title">${c.title}</div>
                    <div class="card-desc">${c.messages ? c.messages.length + ' 条消息' : '0 条消息'}</div>
                </div>
                <button class="btn btn-ghost btn-icon btn-sm" onclick="event.stopPropagation(); deleteConversation('${c.id}')">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                </button>
            </div>
        </div>
    `).join('');
}

async function deleteConversation(id) {
    if (!confirm('确定要删除这个对话吗？')) return;
    try {
        const res = await fetch('/api/conversation/' + id, { method: 'DELETE' });
        const data = await res.json();
        if (data.success) {
            showToast('success', '删除成功', '对话已删除');
            loadConversations();
        }
    } catch (e) {
        showToast('error', '删除失败', '网络错误');
    }
}

// ===== Utilities =====
function parseMarkdown(text) {
    if (!text) return '';
    let html = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    html = html.replace(/__([^_]+)__/g, '<strong>$1</strong>');
    html = html.replace(/_([^_]+)_/g, '<em>$1</em>');
    html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');
    html = html.replace(/^---$/gm, '<hr>');
    html = html.replace(/^[-*+] (.+)$/gm, '<li>$1</li>');
    html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
    html = html.replace(/\n\n+/g, '</p><p>');
    html = html.replace(/\n/g, '<br>');

    return html;
}

function showToast(type, title, message) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    const icons = { success: '✅', error: '❌', warning: '⚠️' };

    toast.innerHTML = `
        <div class="toast-icon">${icons[type] || 'ℹ️'}</div>
        <div class="toast-content">
            <div class="toast-title">${title}</div>
            <div class="toast-message">${message}</div>
        </div>
    `;

    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showSettings() {
    showToast('info', '设置', '设置功能开发中...');
}

// Close modal on overlay click
document.querySelectorAll('.modal-overlay').forEach(overlay => {
    overlay.addEventListener('click', e => {
        if (e.target === overlay) overlay.classList.remove('active');
    });
});