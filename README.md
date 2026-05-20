# 🎓 智慧知识库问答系统

<p align="center">
  <img src="https://img.shields.io/badge/Python-Flask-blue?style=flat&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/RAG-Retrieval-green?style=flat" alt="RAG">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat" alt="License">
</p>

> 基于 RAG（检索增强生成）技术的智能知识库问答管理系统 | [在线预览](https://huomingyao.github.io/personal_rag/templates/index_new.html)

---

## ✨ 功能特性

### 📚 知识库管理
| 功能 | 说明 |
|------|------|
| 🔨 创建知识库 | 支持创建多个独立知识库 |
| 📤 文件上传 | 支持 TXT、PDF、DOCX、Markdown、JSON |
| 🏗️ 向量构建 | 自动构建向量数据库 |
| 🗑️ 删除知识库 | 完整删除数据和向量 |

### 💬 智能问答
| 功能 | 说明 |
|------|------|
| 🔍 语义检索 | 基于向量相似度检索 |
| ✏️ Query改写 | 自动优化检索词 |
| 🌐 联网搜索 | DuckDuckGo补充知识 |
| 🧩 知识块选择 | 手动选择相关知识块 |
| 💾 对话历史 | 自动保存对话记录 |

### 🎨 用户界面
- 📱 响应式布局
- ⚡ SSE实时进度
- 🎯 双页面设计

---

## 🛠️ 技术栈

```
后端     │  Python + Flask
向量库   │  FAISS
嵌入模型 │  BGE (bge-small-zh-v1.5)
大模型   │  MiniMax API
前端     │  HTML + CSS + JS
```

---

## 📁 项目结构

```
personal_rag/
├── rag.py                    # 🚀 入口文件
├── start_server.py           # ⚡ gevent服务器
├── requirements.txt           # 📦 依赖列表
├── README.md                 # 📖 文档
├── src/
│   ├── config.py            # ⚙️  配置模块
│   ├── utils.py           # 🔧  工具函数
│   ├── core.py           # 🧠  核心逻辑
│   └── routes.py        # 🌐  Flask路由
└── templates/
    └── index_new.html   # 🎨  前端页面
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# Windows
set MINIMAX_API=your_api_key

# Linux/Mac
export MINIMAX_API=your_api_key
```

### 3. 启动服务

```bash
# 方式1：直接运行
python rag.py

# 方式2：gevent高并发
python start_server.py
```

### 4. 访问

🌐 http://127.0.0.1:5000

---

## 📡 API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/kb/list` | GET | 获取知识库列表 |
| `/api/kb/create` | POST | 创建知识库 |
| `/api/kb/upload` | POST | 上传文件 |
| `/api/kb/delete` | POST | 删除知识库 |
| `/api/retrieve` | POST | 知识检索 |
| `/api/chat` | POST | 问答对话 |
| `/api/history` | GET | 对话历史 |
| `/build_progress/<kb>` | GET | 构建进度(SSE) |

---

## ⚙️ 配置说明

在 `src/config.py` 中修改：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CHUNK_SIZE` | 500 | 文本分块大小 |
| `CHUNK_OVERLAP` | 50 | 分块重叠 |
| `MODEL` | MiniMax-M2.7 | 大模型 |
| `EMBEDDING_MODEL` | BAAI/bge-small-zh-v1.5 | Embeddings |

---

## 📋 依赖列表

```
flask
langchain-community
langchain-huggingface
langchain-text-splitters
langchain-core
pymupdf
python-docx
openai
duckduckgo-search
gevent
faiss-cpu
```

---

## 🌟 技术亮点

1. **✨ Query改写** - 大模型优化检索词
2. **🔢 多知识库** - 同时检索多个知识库
3. **📡 SSE进度** - 实时显示构建进度
4. **💾 向量缓存** - 避免重复加载
5. **🔌 设备自适应** - 自动检测CUDA/CPU

---

## 📖 使用流程

```
1. 创建知识库 → 2. 上传文档 → 3. 构建向量库 → 4. 智能问答
```

### 创建知识库
1. 输入知识库名称
2. 点击「+」按钮

### 上传文档
1. 选中知识库
2. 上传文件

### 构建向量库
1. 点击「构建」按钮
2. 等待完成

### 智能问答
1. 选择知识库
2. 输入问题
3. 查看结果