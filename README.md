# 智慧知识库问答系统

一个基于 RAG（检索增强生成）技术的智能知识库问答管理系统。

## 在线预览

🔗 **在线预览地址**: https://huomingyao.github.io/personal_rag/templates/index_new.html

> 部署在GitHub Pages上的HTML预览

## 功能简介

### 1. 知识库管理
- **创建知识库** - 创建多个独立的知识库，如课程资料、教师简介、学校通知等
- **文件上传** - 支持上传 TXT、PDF、Word、Markdown、JSON 等格式的文档
- **构建向量库** - 将上传的文档自动构建为向量数据库，支持语义检索
- **删除知识库** - 删除不需要的知识库及其所有文件

### 2. 智能问答
- **多知识库选择** - 可以同时选择多个知识库进行问答
- **语义检索** - 基于向量相似度检索相关内容
- **Query改写** - 自动优化用户问题为更适合检索的形式
- **联网搜索** - 支持DuckDuckGo联网补充知识
- **知识块选择** - 手动选择或取消选择检索到的知识块
- **生成回答** - 使用选中的知识块生成准确回答
- **对话历史** - 自动保存对话记录

### 3. 用户界面
- **双页面设计** - 管理知识库 / 智能问答两个独立页面
- **侧边导航** - 快速切换不同功能模块
- **响应式布局** - 支持桌面端和移动端访问
- **SSE进度** - 实时显示知识库构建进度

## 技术栈

- **后端**: Python + Flask
- **向量库**: FAISS
- **嵌入模型**: HuggingFace Embeddings (BGE)
- **前端**: HTML + CSS + JavaScript
- **大模型**: MiniMax API

## 项目结构

```
d:\api\
├── rag.py              # 入口文件
├── start_server.py     # gevent服务器启动
├── requirements.txt   # Python依赖
├── README.md          # 项目文档
├── src/               # 源代码模块
│   ├── config.py      # 配置模块
│   ├── utils.py      # 工具函数
│   ├── core.py       # 核心逻辑(RAG/问答)
│   └── routes.py    # Flask路由
├── templates/         # 前端模板
│   └── index_new.html
├── multi_knowledge_bases/  # 知识库文件存储
└── multi_vector_bases/     # 向量库存储
```

## 环境要求

- Python 3.10+
- MINIMAX_API_KEY 环境变量

## 安装运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置API密钥
set MINIMAX_API=your_api_key

# 3. 运行
python rag.py

# 或使用gevent支持更高并发
python start_server.py
```

访问 http://127.0.0.1:5000

## 核心模块说明

### src/config.py - 配置模块
```python
# 路径配置
ROOT_KNOWLEDGE_DIR = "multi_knowledge_bases"
ROOT_VECTOR_DIR = "multi_vector_bases"

# 文本分割配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# 大模型配置
MODEL = "MiniMax-M2.7"
BASE_URL = "https://api.minimax.chat/v1"

# Embeddings配置
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
```

### src/utils.py - 工具函数
- `allowed_file()` - 检查文件类型
- `safe_filename()` - 安全文件名处理
- `get_kb_path()` - 获取知识库路径
- `get_all_knowledge_bases()` - 获取所有知识库
- `get_kb_files()` - 获取知识库文件
- `get_all_conversations()` - 获取对话历史
- `save_all_conversations()` - 保存对话历史
- `delete_knowledge_base()` - 删除知识库

### src/core.py - 核心功能
- `init_embeddings()` - 初始化Embeddings模型
- `load_all_files()` - 加载文档(TXT/PDF/DOCX/MD/JSON)
- `rewrite_query()` - Query改写(优化检索词)
- `web_search()` - 联网搜索(DuckDuckGo)
- `build_knowledge_base_generator()` - 向量库构建(SSE进度)
- `retrieve_multi_knowledge()` - 多知识库检索
- `generate_answer()` - 回答生成

### src/routes.py - Flask路由
- 知识库管理API
- 文件上传API
- ���索API
- 问答API
- 对话历史API

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 首页 |
| `/api/kb/list` | GET | 获取知识库列表 |
| `/api/kb/create` | POST | 创建知识库 |
| `/api/kb/upload` | POST | 上传文件 |
| `/api/kb/files` | GET | 获取知识库文件 |
| `/api/kb/delete` | POST | 删除知识库 |
| `/build_progress/<kb>` | GET | 构建进度(SSE) |
| `/api/retrieve` | POST | 知识检索 |
| `/api/chat` | POST | 问答对话 |
| `/api/history` | GET | 对话历史 |
| `/api/conversation` | POST | 创建对话 |
| `/api/conversation/<id>` | GET/PUT/DELETE | 对话操作 |
| `/api/conversation/<id>/message` | POST | 保存消息 |

## 使用流程

### 1. 创建知识库
1. 点击「知识库管理」页面
2. 在输入框中输入知识库名称（如：课程资料）
3. 点击「+」按钮创建

### 2. 上传文档
1. 点击要上传的知识库卡片选中它
2. 点击上传区域或拖拽文件到上传区域
3. 等待上传完成

### 3. 构建向量库
1. 点击知识库卡片上的「构建」按钮
2. 等待构建完成（状态变为「已构建」）

### 4. 智能问答
1. 切换到「智能问答」页面
2. 在左侧选择要使用的知识库
3. 输入问题并点击「提问」
4. 查看检索到的知识块
5. 可选择相关知识块后点击生成回答

## 问答示例

```
用户: 什么是深度学习?
系统: [从知识库检索相关内容]
回答: 深度学习是机器学习的一个分支，它使用多层神经网络...
```

## 配置修改

可在 `src/config.py` 中修改：

```python
CHUNK_SIZE = 500        # 文本分块大小
CHUNK_OVERLAP = 50      # 分块重叠大小
MODEL = "MiniMax-M2.7"   # 大模型名称
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"  # Embeddings模型
```

## 依赖列表

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

## 技术亮点

1. **Query改写** - 使用大模型将用户问题改写为更适合检索的形式
2. **多知识库** - 支持同时检索多个知识库
3. **SSE进度** - 知识库构建进度实时推送
4. **向量缓存** - 避免重复加载向量库
5. **设备自适应** - 自动检测CUDA/CPU