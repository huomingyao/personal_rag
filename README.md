# 智慧知识库问答系统

一个基于 RAG（检索增强生成）技术的智能知识库问答管理系统。

## 功能简介

### 1. 知识库管理
- **创建知识库** - 创建多个独立的知识库，如课程资料、教师简介、学校通知等
- **文件上传** - 支持上传 TXT、PDF、Word、Markdown、JSON 等格式的文档
- **构建向量库** - 将上传的文档自动构建为向量数据库，支持语义检索
- **删除知识库** - 删除不需要的知识库及其所有文件

### 2. 智能问答
- **多知识库选择** - 可以同时选择多个知识库进行问答
- **语义检索** - 基于向量相似度检索相关内容
- **知识块选择** - 手动选择或取消选择检索到的知识块
- **生成回答** - 使用选中的知识块生成准确回答

### 3. 用户界面
- **双页面设计** - 管理知识库 / 智能问答两个独立页面
- **侧边导航** - 快速切换不同功能模块
- **响应式布局** - 支持桌面端和移动端访问

## 技术栈

- **后端**: Python + Flask
- **向量库**: FAISS
- **嵌入模型**: sentence-transformers
- **前端**: HTML + CSS + JavaScript

## 安装依赖

```bash
pip install flask flask-cors flask-uploads
pip install langchain langchain-community
pip install faiss-cpu sentence-transformers
pip install python-magic pdfplumber
```

## 启动服务

```bash
python start_server.py
```

服务启动后访问：
- 本地: http://127.0.0.1:5000
- 局域网: http://本机IP:5000

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
4. 选择相关的知识块
5. 点击「使用选中的知识块生成回答」

## 项目结构

```
d:\api
├── rag.py                  # Flask 后端主程序
├── start_server.py         # 服务启动脚本
├── templates/
│   └── index_new.html     # 前端页面
├── knowledge/            # 知识库原始文件存储
├── vectors/             # 向量索引存储
└── README.md            # 本文件
```

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 首页 |
| `/api/kb/list` | GET | 获取知识库列表 |
| `/api/kb/create` | POST | 创建知识库 |
| `/api/kb/upload` | POST | 上传文件 |
| `/api/kb/files` | GET | 获取知识库文件列表 |
| `/api/kb/delete` | POST | 删除知识库 |
| `/chat` | POST | 问答接口 |