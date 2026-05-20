"""工具函数模块 - 辅助函数和工具方法"""

import os
import re
import json
import time
import shutil
from typing import List, Dict, Optional
from urllib.parse import unquote

from .config import ROOT_KNOWLEDGE_DIR, ROOT_VECTOR_DIR, ALLOWED_EXTENSIONS


def allowed_file(filename: str) -> bool:
    """检查文件类型是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def safe_filename(name: str) -> str:
    """安全的文件名处理，保留中文字符"""
    name = name.strip()
    name = re.sub(r'[\\/:*?"<>|]', '_', name)
    if len(name) > 100:
        name = name[:100]
    return name


def get_kb_path(kb_name: str, kb_type: str = "knowledge") -> str:
    """获取知识库/向量库路径"""
    safe_name = safe_filename(kb_name)
    if kb_type == "knowledge":
        return os.path.join(ROOT_KNOWLEDGE_DIR, safe_name)
    elif kb_type == "vector":
        return os.path.join(ROOT_VECTOR_DIR, safe_name)
    raise ValueError("type must be 'knowledge' or 'vector'")


def get_all_knowledge_bases() -> List[Dict]:
    """获取所有知识库列表"""
    kb_list = []
    if os.path.exists(ROOT_KNOWLEDGE_DIR):
        for kb_name in os.listdir(ROOT_KNOWLEDGE_DIR):
            kb_path = os.path.join(ROOT_KNOWLEDGE_DIR, kb_name)
            if os.path.isdir(kb_path):
                file_count = sum([len(files) for _, _, files in os.walk(kb_path)])
                vector_path = os.path.join(ROOT_VECTOR_DIR, kb_name)
                is_built = os.path.exists(vector_path)
                kb_list.append({
                    "name": kb_name,
                    "file_count": file_count,
                    "is_built": is_built,
                    "knowledge_path": kb_path,
                    "vector_path": vector_path
                })
    return kb_list


def get_kb_files(kb_name: str) -> List[Dict]:
    """获取指定知识库中的所有文件列表"""
    files = []
    kb_path = get_kb_path(kb_name, "knowledge")
    if os.path.exists(kb_path):
        for root, _, filenames in os.walk(kb_path):
            for filename in filenames:
                if '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, kb_path)
                    files.append({
                        "name": filename,
                        "path": rel_path,
                        "size": os.path.getsize(file_path),
                        "modified": os.path.getmtime(file_path)
                    })
    return sorted(files, key=lambda x: x["modified"], reverse=True)


def get_all_conversations() -> List[Dict]:
    """获取所有对话"""
    from .config import HISTORY_FILE
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('conversations', [])
        except:
            pass
    return []


def save_all_conversations(conversations: List[Dict]) -> None:
    """保存所有对话"""
    from .config import HISTORY_FILE
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump({'conversations': conversations}, f, ensure_ascii=False, indent=2)


def delete_knowledge_base(kb_name: str) -> bool:
    """删除知识库（含文件和向量库）"""
    try:
        kb_path = get_kb_path(kb_name, "knowledge")
        if os.path.exists(kb_path):
            shutil.rmtree(kb_path)
        vector_path = get_kb_path(kb_name, "vector")
        if os.path.exists(vector_path):
            shutil.rmtree(vector_path)
        return True
    except Exception as e:
        print(f"[ERROR] 删除知识库失败: {e}")
        return False


def clean_directory(directory: str) -> bool:
    """清理目录"""
    try:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        print(f"[ERROR] 清理目录失败: {e}")
        return False