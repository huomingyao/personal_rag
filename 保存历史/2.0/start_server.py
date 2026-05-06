# start_server.py 修复版
from gevent import pywsgi
from rag import app  # 导入Flask应用（已重命名为rag.py）

if __name__ == "__main__":
    # 🔥 修复：正确参数 listener，同时适配gevent所有版本
    server = pywsgi.WSGIServer(('0.0.0.0', 5000), app)
    
    print("===== 本地知识库问答系统 - 多线程版 =====")
    print("🌐 本地访问：http://127.0.0.1:5000")
    print("🌐 局域网访问：http://本机IP:5000")
    print("🔧 按 Ctrl+C 停止服务")
    print("======================================\n")
    
    # 启动服务器（支持数十人并发访问）
    server.serve_forever()