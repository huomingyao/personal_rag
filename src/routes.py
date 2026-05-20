"""Flask路由模块 - 所有API路由"""

import os
import time
import uuid
from flask import (
    Blueprint, render_template, request, jsonify, Response, stream_with_context
)

from .config import MAX_CONTENT_LENGTH
from .utils import (
    allowed_file, safe_filename, get_kb_path, get_all_knowledge_bases, get_kb_files,
    get_all_conversations, save_all_conversations, delete_knowledge_base
)
from .core import build_knowledge_base_generator, retrieve_multi_knowledge, generate_answer

# 创建Blueprint
api = Blueprint('api', __name__)

# ===================== 页面路由 =====================
@api.route('/')
def index():
    """主页"""
    return render_template('/templates/index_new.html')


# ===================== 知识库API =====================
@api.route('/api/kb/list')
def api_kb_list():
    """获取所有知识库列表"""
    try:
        kb_list = get_all_knowledge_bases()
        return jsonify({"success": True, "data": kb_list})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@api.route('/api/kb/create', methods=['POST'])
def api_kb_create():
    """创建新知识库"""
    try:
        data = request.get_json()
        kb_name = data.get('name', '').strip()

        if not kb_name:
            return jsonify({"success": False, "message": "知识库名称不能为空"})

        kb_path = get_kb_path(kb_name, "knowledge")
        if os.path.exists(kb_path):
            return jsonify({"success": False, "message": f"知识库[{kb_name}]已存在"})

        os.makedirs(kb_path)
        return jsonify({"success": True, "message": f"知识库[{kb_name}]创建成功"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@api.route('/api/kb/upload', methods=['POST'])
def api_kb_upload():
    """上传文件到指定知识库"""
    try:
        kb_name = request.form.get('kb_name', '').strip()
        if not kb_name:
            return jsonify({"success": False, "message": "请选择要上传的知识库"})

        if 'file' not in request.files:
            return jsonify({"success": False, "message": "请选择要上传的文件"})

        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "message": "文件名称不能为空"})

        if file and allowed_file(file.filename):
            kb_path = get_kb_path(kb_name, "knowledge")
            filename = safe_filename(file.filename)
            file_path = os.path.join(kb_path, filename)
            file.save(file_path)

            return jsonify({
                "success": True,
                "message": f"文件[{filename}]上传成功",
                "filename": filename,
                "auto_build": False,
                "kb_name": kb_name
            })
        else:
            return jsonify({"success": False, "message": "仅支持上传txt/pdf/docx/md/json文件"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@api.route('/api/kb/files', methods=['GET'])
def api_kb_files():
    """获取指定知识库的文件列表"""
    try:
        kb_name = request.args.get('name', '').strip()
        if not kb_name:
            return jsonify({"success": False, "message": "知识库名称不能为空"})

        files = get_kb_files(kb_name)
        return jsonify({"success": True, "data": files})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@api.route('/api/kb/delete', methods=['POST'])
def api_kb_delete():
    """删除指定知识库"""
    try:
        data = request.get_json()
        kb_name = data.get('name', '').strip()

        if not kb_name:
            return jsonify({"success": False, "message": "知识库名称不能为空"})

        if delete_knowledge_base(kb_name):
            return jsonify({"success": True, "message": f"知识库[{kb_name}]删除成功"})
        return jsonify({"success": False, "message": "删除失败"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


# ===================== 构建进度SSE =====================
@api.route('/build_progress/<path:kb_name>')
def build_progress(kb_name):
    """SSE路由：返回指定知识库的构建进度"""
    from urllib.parse import unquote
    kb_name = unquote(kb_name)

    @stream_with_context
    def generate():
        import json
        for item in build_knowledge_base_generator(kb_name):
            yield f"data: {json.dumps(item)}\n\n"
            if item.get("progress") == 100:
                break
    return Response(generate(), mimetype='text/event-stream')


# ===================== 检索API =====================
@api.route('/api/retrieve', methods=['POST'])
def api_retrieve():
    """只进行知识检索，返回切片结果"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        selected_kbs = data.get('selected_kbs', [])

        if not question:
            return jsonify({"error": "请输入问题"}), 400
        if not selected_kbs:
            return jsonify({"error": "请选择要检索的知识库"}), 400

        all_local_knowledge, original_question, rewritten_query = retrieve_multi_knowledge(
            question, selected_kbs
        )

        return jsonify({
            "success": True,
            "question": original_question,
            "rewritten_query": rewritten_query,
            "knowledge_chunks": all_local_knowledge
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===================== 问答API =====================
@api.route('/api/chat', methods=['POST'])
def api_chat():
    """使用选中的知识切片生成回答"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        selected_kbs = data.get('selected_kbs', [])
        selected_knowledge = data.get('selected_knowledge', [])
        enable_web = data.get('enable_web', False)

        if not question:
            return jsonify({"error": "请输入问题"}), 400
        if not selected_kbs:
            return jsonify({"error": "请选择要检索的知识库"}), 400

        result = generate_answer(question, selected_kbs, selected_knowledge, enable_web)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===================== 对话历史API =====================
@api.route('/api/history', methods=['GET'])
def api_get_history():
    """获取所有对话列表"""
    try:
        conversations = get_all_conversations()
        return jsonify({"success": True, "data": conversations})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@api.route('/api/conversation', methods=['POST'])
def api_create_conversation():
    """创建新对话"""
    try:
        data = request.get_json()
        title = data.get('title', '新对话').strip()

        conversations = get_all_conversations()
        new_id = str(uuid.uuid4())[:8]

        new_conv = {
            "id": new_id,
            "title": title,
            "messages": [],
            "time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        conversations.insert(0, new_conv)
        save_all_conversations(conversations)

        return jsonify({"success": True, "data": new_conv})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@api.route('/api/conversation/<conv_id>', methods=['GET'])
def api_get_conversation(conv_id):
    """获取指定对话的详情"""
    try:
        conversations = get_all_conversations()
        conv = next((c for c in conversations if c.get('id') == conv_id), None)
        if conv:
            return jsonify({"success": True, "data": conv})
        return jsonify({"success": False, "message": "对话不存在"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@api.route('/api/conversation/<conv_id>', methods=['PUT'])
def api_update_conversation(conv_id):
    """更新对话信息"""
    try:
        data = request.get_json()
        title = data.get('title', '').strip()

        conversations = get_all_conversations()
        conv = next((c for c in conversations if c.get('id') == conv_id), None)

        if not conv:
            return jsonify({"success": False, "message": "对话不存在"})

        if title:
            conv['title'] = title

        save_all_conversations(conversations)
        return jsonify({"success": True, "data": conv})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@api.route('/api/conversation/<conv_id>', methods=['DELETE'])
def api_delete_conversation(conv_id):
    """删除指定对话"""
    try:
        conversations = get_all_conversations()
        conversations = [c for c in conversations if c.get('id') != conv_id]
        save_all_conversations(conversations)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@api.route('/api/conversation/<conv_id>/message', methods=['POST'])
def api_save_message(conv_id):
    """保存对话消息"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        answer = data.get('answer', '').strip()
        thinking = data.get('thinking', '').strip()

        if not question:
            return jsonify({"success": False, "message": "问题不能为空"})

        conversations = get_all_conversations()
        conv = next((c for c in conversations if c.get('id') == conv_id), None)

        if not conv:
            return jsonify({"success": False, "message": "对话不存在"})

        conv['messages'].insert(0, {
            "question": question,
            "answer": answer,
            "thinking": thinking,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        if len(conv['messages']) == 1:
            conv['title'] = question[:30] + ('...' if len(question) > 30 else '')

        conv['time'] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_all_conversations(conversations)

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})