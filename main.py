#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视频分析系统 - FastAPI 接口版
部署到 PythonAnywhere / 云端
接口：/analyze_video
支持：异步任务 + 任务ID查询
"""

import os
import json
import time
import uuid
import asyncio
from datetime import datetime
from typing import Dict, Optional, Any
import threading

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse

# 导入你的工具函数
from tools import convert_to_serializable, ensure_dir, format_duration

# 导入分析模块
from video_smoothness import VideoSmoothnessAnalyzer
from gait_face_analyzer import GaitFaceAnalyzer
from audio_analyzer import AudioAnalyzer, AudioAnalyzerSimple
from sensitive_info_detector import SensitiveInfoDetector, PrivacyBlur

# ==============================================
# 全局配置（云端必须关闭打印、禁用GUI）
# ==============================================
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
cv2.setLogLevel(0)

# 全局任务存储（云端简易版，正式用Redis）
TASKS = {}

# FastAPI 应用
app = FastAPI(title="视频步态+人脸分析API", version="2.0")

# 全局分析器实例
analyzer_app = VideoAnalysisApp(
    enable_audio=True,
    use_advanced_asr=False,
    enable_sensitive=True
)

# ==============================================
# 核心：视频分析类（保持你原有逻辑不变）
# ==============================================
class VideoAnalysisApp:
    def __init__(self, enable_audio: bool = True, use_advanced_asr: bool = False,
                 enable_sensitive: bool = True, ffmpeg_path: str = None):
        self.smoothness_analyzer = VideoSmoothnessAnalyzer()
        self.gait_face_analyzer = GaitFaceAnalyzer()
        self.audio_analyzer = None
        if enable_audio:
            self.audio_analyzer = AudioAnalyzerSimple(ffmpeg_path=ffmpeg_path)
        self.sensitive_detector = SensitiveInfoDetector() if enable_sensitive else None
        self.privacy_blur = PrivacyBlur() if enable_sensitive else None

    def reset(self):
        self.smoothness_analyzer.reset()
        self.gait_face_analyzer.reset()

    def analyze_video(self, video_path: str, task_id: str) -> Dict:
        try:
            self.reset()
            if not os.path.exists(video_path):
                return {"status": "error", "message": f"文件不存在"}

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"status": "error", "message": "无法打开视频"}

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                self.smoothness_analyzer.evaluate_frame(frame, time.time())
                self.gait_face_analyzer.process_frame(frame)

                # 更新进度
                progress = round(frame_count / total_frames * 100, 1)
                TASKS[task_id]["progress"] = progress

            cap.release()

            smoothness_stats = self.smoothness_analyzer.get_final_stats()
            gait_face_stats = self.gait_face_analyzer.get_final_stats(total_frames, fps)
            audio_result = {}
            sensitive_result = {}

            result = {
                "status": "success",
                "video_info": {
                    "file_name": os.path.basename(video_path),
                    "width": width,
                    "height": height,
                    "fps": round(fps, 2),
                    "duration_seconds": round(duration, 2),
                    "duration_formatted": format_duration(duration)
                },
                "smoothness_analysis": smoothness_stats,
                "gait_analysis": gait_face_stats["gait_analysis"],
                "face_detection": gait_face_stats["face_detection"],
                "audio_analysis": audio_result,
                "sensitive_detection": sensitive_result,
            }
            return result

        except Exception as e:
            return {"status": "error", "message": str(e)}

# ==============================================
# 接口 1：上传视频并开始分析（异步任务）
# ==============================================
@app.post("/analyze_video")
async def analyze_video(
    file: UploadFile = File(...),
):
    # 生成唯一任务ID
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {
        "status": "running",
        "progress": 0.0,
        "result": None
    }

    # 保存临时文件
    ensure_dir("tmp")
    video_path = f"tmp/{task_id}_{file.filename}"

    with open(video_path, "wb") as f:
        f.write(await file.read())

    # 后台线程执行分析（避免超时）
    def task():
        res = analyzer_app.analyze_video(video_path, task_id)
        TASKS[task_id]["result"] = res
        TASKS[task_id]["status"] = "completed"

    threading.Thread(target=task, daemon=True).start()

    return {
        "task_id": task_id,
        "status": "running",
        "progress_url": f"/task/{task_id}"
    }

# ==============================================
# 接口 2：查询任务进度/结果
# ==============================================
@app.get("/task/{task_id}")
def get_task_status(task_id: str):
    if task_id not in TASKS:
        return JSONResponse({"status": "error", "message": "任务不存在"}, 404)
    return TASKS[task_id]

# ==============================================
# 健康检查（PythonAnywhere 需要）
# ==============================================
@app.get("/")
def home():
    return {"status": "ok", "message": "视频分析API运行正常"}

# ==============================================
# WSGI 入口（PythonAnywhere 必须）
# ==============================================
application = app