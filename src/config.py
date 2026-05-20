"""配置文件 - 集中管理所有配置"""

import os
import warnings

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_CACHE'] = os.path.join(os.getcwd(), 'models')
warnings.filterwarnings("ignore")

# ===================== 路径配置 =====================
ROOT_KNOWLEDGE_DIR = os.path.join(os.getcwd(), "multi_knowledge_bases")
ROOT_VECTOR_DIR = os.path.join(os.getcwd(), "multi_vector_bases")
HISTORY_FILE = os.path.join(os.getcwd(), 'chat_history.json')

# 确保目录存在
os.makedirs(ROOT_KNOWLEDGE_DIR, exist_ok=True)
os.makedirs(ROOT_VECTOR_DIR, exist_ok=True)

# ===================== 文件类型配置 =====================
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'md', 'json'}

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ===================== 大模型配置 =====================
API_KEY = os.environ.get("MINIMAX_API")
MODEL = "MiniMax-M2.7"
BASE_URL = "https://api.minimax.chat/v1"

# ===================== 设备配置 =====================
def check_device():
    """检测可用设备"""
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda', torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return 'cpu', None

DEVICE, GPU_NAME = check_device()

# ===================== Embeddings模型配置 =====================
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# ===================== Flask配置 =====================
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB