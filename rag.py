"""主入口文件 - Flask应用启动"""

import os
import sys

# 添加src目录到路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from flask import Flask
from src.config import (
    ROOT_KNOWLEDGE_DIR, ROOT_VECTOR_DIR, MAX_CONTENT_LENGTH,
    FLASK_HOST, FLASK_PORT
)
from src.utils import get_all_knowledge_bases, get_kb_path, safe_filename
from src.routes import api

# ===================== 检查依赖 =====================
LANGCHAIN_AVAILABLE = True
LANGCHAIN_ERROR = ""

try:
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    LANGCHAIN_ERROR = str(e)
    print(f"[ERROR] 缺少必要的依赖: {e}")
    print("[INFO] 请运行以下命令安装依赖:")
    print("  pip install langchain-community langchain-huggingface langchain-text-splitters")

# ===================== Flask应用初始化 =====================
app = Flask(__name__, template_folder='.')
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# 注册Blueprint
app.register_blueprint(api)


# ===================== 主程序 =====================
if __name__ == "__main__":
    if not LANGCHAIN_AVAILABLE:
        print(f"[ERROR] 缺少必要依赖: {LANGCHAIN_ERROR}")
        print("[INFO] 请运行: pip install langchain-community langchain-huggingface langchain-text-splitters pymupdf python-docx")
        exit(1)

    from src.config import API_KEY, DEVICE, GPU_NAME
    if not API_KEY:
        print("[ERROR] 未配置MINIMAX_API环境变量，请先配置！")
        exit(1)

    print("===== 多知识库问答系统 =====")
    print(f"🖥️  检测到设备: {DEVICE}" + (f" ({GPU_NAME})" if GPU_NAME else ""))
    print("🌐 网页访问地址：http://127.0.0.1:5000")
    print("🔧 按 Ctrl+C 停止服务")
    print("======================================\n")

    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=False,
        threaded=True
    )