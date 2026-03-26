#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视频分析系统 - 阿里云轻量服务器 + 完整JSON结果返回
修复：最终分析结果JSON从接口正常返回
"""

import os
import time
import uuid
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

# ====================== 启动 ======================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)