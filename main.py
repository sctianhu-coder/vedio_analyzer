#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视频分析系统 - 阿里云轻量服务器 + 简易日志追踪版
"""

import os
import time
import uuid
import threading
from datetime import datetime
from typing import Dict

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse 
from fastapi.staticfiles import StaticFiles  # 添加这一行

# 工具导入
from tools import ensure_dir, format_duration

# 分析模块
from video_smoothness import VideoSmoothnessAnalyzer
from gait_face_analyzer import GaitFaceAnalyzer
from audio_analyzer import AudioAnalyzerSimple
from sensitive_info_detector import SensitiveInfoDetector, PrivacyBlur

# 全局任务
TASKS = {}
app = FastAPI(title="视频分析API", version="simple-log")

app = FastAPI(title="视频分析API", version="simple-log")

# ====================== 添加静态文件服务 ======================
# 确保 static 目录存在
os.makedirs("static", exist_ok=True)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def home():
    """返回前端页面"""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return {"status": "error", "msg": "前端页面未找到，请将 index.html 放入 static 目录"}


# ====================== 简单日志工具 ======================
def log(task_id, msg):
    """统一日志格式：时间 + 任务ID + 信息"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] [任务:{task_id}] {msg}")

# ====================== 视频分析类 ======================
class VideoAnalysisApp:
    def __init__(self):
        self.smoothness_analyzer = VideoSmoothnessAnalyzer()
        self.gait_face_analyzer = GaitFaceAnalyzer()
        self.audio_analyzer = AudioAnalyzerSimple()
        self.sensitive_detector = SensitiveInfoDetector()

    def analyze_video(self, video_path, task_id):
        try:
            log(task_id, f"开始处理，文件：{os.path.basename(video_path)}")

            # 打开视频
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                log(task_id, "错误：无法打开视频")
                return {"status": "error", "message": "无法打开视频"}

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            log(task_id, f"视频总帧数：{total_frames}")

            # 逐帧处理
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                self.smoothness_analyzer.evaluate_frame(frame, time.time())
                self.gait_face_analyzer.process_frame(frame)

                # 每30帧更新进度
                if frame_count % 30 == 0:
                    progress = round(frame_count / total_frames * 100, 1)
                    TASKS[task_id]["progress"] = progress
                    log(task_id, f"处理进度：{progress}%")

            cap.release()

            # 结果
            smooth = self.smoothness_analyzer.get_final_stats()
            gait_face = self.gait_face_analyzer.get_final_stats(total_frames, cap.get(cv2.CAP_PROP_FPS))
            log(task_id, "处理完成！")

            return {
                "status": "success",
                "progress": 100,
                "smoothness": smooth,
                "face": gait_face["face_detection"]
            }

        except Exception as e:
            log(task_id, f"处理失败：{str(e)}")
            return {"status": "error", "message": str(e)}

# ====================== 全局实例 ======================
analyzer_app = VideoAnalysisApp()

# ====================== API 接口 ======================
@app.post("/analyze_video")
async def analyze_video(file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())[:8]  # 短ID，方便看日志
    filename = file.filename

    # 日志
    log(task_id, f"收到上传文件：{filename}")

    # 初始化任务
    TASKS[task_id] = {
        "status": "running",
        "progress": 0,
        "file": filename,
        "result": None
    }

    # 保存文件
    ensure_dir("tmp")
    video_path = f"tmp/{task_id}_{filename}"

    with open(video_path, "wb") as f:
        f.write(await file.read())

    log(task_id, "文件保存完成，开始后台处理")

    # 后台处理
    def run():
        res = analyzer_app.analyze_video(video_path, task_id)
        TASKS[task_id]["result"] = res
        TASKS[task_id]["status"] = "completed"

    threading.Thread(target=run, daemon=True).start()
    return {"task_id": task_id, "status": "接收成功，处理中"}

@app.get("/task/{task_id}")
def get_task(task_id: str):
    if task_id not in TASKS:
        return {"status": "error", "msg": "不存在"}
    return TASKS[task_id]

@app.get("/")
def home():
    return {"status": "ok", "msg": "视频分析API运行正常"}

# ====================== 启动 ======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)