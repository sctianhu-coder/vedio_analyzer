#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视频分析系统 - API服务
保留完整JSON返回、任务管理、恶意请求过滤逻辑
"""
import os
import re
import uuid
import logging
import threading
from datetime import datetime
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# 导入核心模块
from core import VideoAnalysisCore
from tools import ensure_dir

# 全局配置
TASKS = {}
app = FastAPI(title="视频分析API", version="final")

# 静态文件
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ====================== 日志配置（保留原逻辑） ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
uvicorn_log = logging.getLogger("uvicorn")
uvicorn_log.handlers = []
uvicorn_log.addHandler(logging.StreamHandler())
uvicorn_log.setLevel(logging.INFO)

uvicorn_access_log = logging.getLogger("uvicorn.access")
uvicorn_access_log.handlers = []
uvicorn_access_log.addHandler(logging.StreamHandler())
uvicorn_access_log.setLevel(logging.INFO)

def log(task_id, msg):
    """任务日志函数"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] [任务:{task_id}] {msg}")

# ====================== 恶意请求过滤（完整保留原逻辑） ======================
# 恶意路径正则规则
MALICIOUS_PATTERNS = [
    # 路径遍历攻击
    r"\.\./",
    r"/etc/(passwd|hosts|shadow)",
    # 敏感配置文件
    r"\.env",
    r"docker-compose\.(yml|yaml)",
    r"Dockerfile",
    r"\.git/",
    r"config\.json",
    # 敏感脚本/源码文件
    r"\.(php|py|go|js)$",
    r"phpinfo\.php",
    r"test\.php",
    # 其他恶意特征
    r"file://",
    r"META-INF",
    r"login\.action",
    r"\.vscode/",
    r"\.secret",
    r"claude",
]
# 编译正则
MALICIOUS_REGEX = re.compile("|".join(MALICIOUS_PATTERNS), re.I)

# 全局拦截中间件
@app.middleware("http")
async def block_malicious_requests(request: Request, call_next):
    path = request.url.path
    query = str(request.query_params)

    # 匹配恶意特征 → 直接拒绝
    if MALICIOUS_REGEX.search(path) or MALICIOUS_REGEX.search(query):
        return JSONResponse(
            status_code=403,
            content={"code": 403, "msg": "Forbidden: Illegal request"}
        )

    # 正常请求放行
    response = await call_next(request)
    return response

# ====================== 核心实例初始化 ======================
analyzer_core = VideoAnalysisCore(
    enable_audio=True,
    use_advanced_asr=False,
    enable_sensitive=True,
    ffmpeg_path=None  # 自动识别
)

# ====================== API接口（保留原逻辑） ======================
@app.post("/analyze_video")
async def analyze_video(file: UploadFile = File(...)):
    """视频分析接口"""
    task_id = str(uuid.uuid4())[:8]
    filename = file.filename
    log(task_id, f"收到文件：{filename}")

    # 初始化任务状态
    TASKS[task_id] = {
        "status": "running",
        "progress": 0,
        "file": filename,
        "result": None
    }

    # 保存上传文件
    ensure_dir("tmp")
    video_path = f"tmp/{task_id}_{filename}"
    with open(video_path, "wb") as f:
        f.write(await file.read())

    # 异步执行分析
    def run_analysis():
        """异步分析函数"""
        def progress_cb(progress: float):
            """进度更新回调"""
            TASKS[task_id]["progress"] = progress

        # 调用核心分析逻辑
        result = analyzer_core.analyze_video(
            video_path=video_path,
            task_id=task_id,
            progress_callback=progress_cb,
            verbose=True,
            analyze_audio=True,
            detect_sensitive=True
        )

        # 更新任务状态
        TASKS[task_id]["result"] = result
        TASKS[task_id]["status"] = "completed"
        TASKS[task_id]["progress"] = 100

    # 启动异步线程
    threading.Thread(target=run_analysis, daemon=True).start()
    return {"task_id": task_id, "status": "running", "progress": 0}

@app.get("/task/{task_id}")
def get_task(task_id: str):
    """查询任务结果接口"""
    if task_id not in TASKS:
        return {"status": "error", "msg": "任务不存在"}

    task = TASKS[task_id]
    if task["status"] == "completed":
        return JSONResponse(content={
            "task_id": task_id,
            "status": "completed",
            "progress": 100,
            "data": task["result"]
        })
    return task

@app.get("/")
async def home():
    """首页"""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except:
        return {"msg": "视频分析API运行正常", "upload": "/analyze_video", "query": "/task/{task_id}"}

# ====================== 启动服务 ======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)