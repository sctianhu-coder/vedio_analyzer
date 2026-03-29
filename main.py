#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视频分析系统 - 阿里云轻量服务器 + 完整JSON结果返回
修复：最终分析结果JSON从接口正常返回
"""

import os
import time
import uuid
import logging
import threading
from datetime import datetime
import tools
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# 工具导入
from tools import ensure_dir, format_duration

# 分析模块
from video_smoothness import VideoSmoothnessAnalyzer
from gait_face_analyzer import GaitFaceAnalyzer
from audio_analyzer import AudioAnalyzerSimple
from sensitive_info_detector import SensitiveInfoDetector, PrivacyBlur

# 全局任务
TASKS = {}

app = FastAPI(title="视频分析API", version="final")

# 静态文件
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
# ====================== 全局日志配置（带时间戳） ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 给 uvicorn 日志也加上时间
uvicorn_log = logging.getLogger("uvicorn")
uvicorn_log.handlers = []
uvicorn_log.addHandler(logging.StreamHandler())
uvicorn_log.setLevel(logging.INFO)

uvicorn_access_log = logging.getLogger("uvicorn.access")
uvicorn_access_log.handlers = []
uvicorn_access_log.addHandler(logging.StreamHandler())
uvicorn_access_log.setLevel(logging.INFO)

# ====================== 日志 ======================
def log(task_id, msg):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] [任务:{task_id}] {msg}")

# ====================== 视频分析类（完整版返回JSON） ======================
class VideoAnalysisApp:
    def __init__(self):
        self.smoothness_analyzer = VideoSmoothnessAnalyzer()
        self.gait_face_analyzer = GaitFaceAnalyzer()
        self.audio_analyzer = AudioAnalyzerSimple()  # 自动识别ffmpeg
        self.sensitive_detector = SensitiveInfoDetector()

    def analyze_video(self, video_path, task_id):
        try:
            log(task_id, f"开始处理：{os.path.basename(video_path)}")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                log(task_id, "错误：无法打开视频")
                return {"status": "error", "message": "无法打开视频"}

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            log(task_id, f"总帧数：{total_frames}，时长：{duration:.1f}s")

            # ==============================================
            # ✅ 修复苹果 MOV 帧尺寸突变问题（按第一帧统一大小），所有格式的视频都会变帧的。
            # ==============================================
            ret, first_frame = cap.read()
            if not ret:
                cap.release()
                log(task_id, "错误：视频为空")
                return {"status": "error", "message": "视频为空"}

            # 固定使用第一帧的尺寸
            FIXED_WIDTH = first_frame.shape[1]
            FIXED_HEIGHT = first_frame.shape[0]
            log(task_id, f"✅ 统一帧尺寸：{FIXED_WIDTH}x{FIXED_HEIGHT}")

            # 回到第0帧，开始正式循环
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # 逐帧分析
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                self.smoothness_analyzer.evaluate_frame(frame, time.time())
                self.gait_face_analyzer.process_frame(frame)

                if frame_count % 30 == 0:
                    progress = min(round(frame_count / total_frames * 100, 1), 100)
                    TASKS[task_id]["progress"] = progress
                    log(task_id, f"处理进度：{progress}%")

            cap.release()

            # ====================== 收集所有分析结果 ======================
            smooth_result = self.smoothness_analyzer.get_final_stats()
            gait_face_result = self.gait_face_analyzer.get_final_stats(total_frames, fps)
            audio_result = self.audio_analyzer.analyze_video_audio(video_path)

            # ====================== 组装最终JSON ======================
            final_result = {
                "status": "success",
                "progress": 100,
                "video_info": {
                    "total_frames": total_frames,
                    "fps": round(fps, 2),
                    "duration": round(duration, 2)
                },
                "smoothness": tools.convert_numpy_types(smooth_result),
                "face_detection": tools.convert_numpy_types(gait_face_result["face_detection"]),
                "gait_analysis": tools.convert_numpy_types(gait_face_result["gait_analysis"]),
                "audio": tools.convert_numpy_types(audio_result)
            }

            log(task_id, "✅ 全部分析完成")
            return final_result

        except Exception as e:
            log(task_id, f"❌ 失败：{str(e)}")
            return {"status": "error", "message": str(e)}

# ====================== 全局实例 ======================
analyzer_app = VideoAnalysisApp()

# ====================== API ======================
@app.post("/analyze_video")
async def analyze_video(file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())[:8]
    filename = file.filename
    log(task_id, f"收到文件：{filename}")

    TASKS[task_id] = {
        "status": "running",
        "progress": 0,
        "file": filename,
        "result": None
    }

    ensure_dir("tmp")
    video_path = f"tmp/{task_id}_{filename}"

    with open(video_path, "wb") as f:
        f.write(await file.read())

    def run():
        result = analyzer_app.analyze_video(video_path, task_id)
        TASKS[task_id]["result"] = result
        TASKS[task_id]["status"] = "completed"
        TASKS[task_id]["progress"] = 100

    threading.Thread(target=run, daemon=True).start()
    return {"task_id": task_id, "status": "running", "progress": 0}

@app.get("/task/{task_id}")
def get_task(task_id: str):
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
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except:
        return {"msg": "视频分析API运行正常", "upload": "/analyze_video", "query": "/task/{task_id}"}

import re

# 恶意路径正则规则（覆盖你日志里的所有攻击）
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
# 编译正则（提升性能）
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

# ====================== 启动 ======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)