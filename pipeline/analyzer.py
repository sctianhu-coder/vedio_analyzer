#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视频分析核心模块 - 使用 Pipeline 架构
"""

import os
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Callable

import tools
from decoder import VideoDecoder
from pipeline import PipelineBuilder, VideoPipeline


class VideoAnalysisCore:
    """视频分析核心 - 使用 Pipeline 架构"""

    def __init__(self,
                 enable_audio: bool = True,
                 use_advanced_asr: bool = False,
                 enable_sensitive: bool = True,
                 max_workers: int = None):
        """
        初始化分析器

        Args:
            enable_audio: 是否启用音频分析
            use_advanced_asr: 是否使用高级语音识别（暂未实现）
            enable_sensitive: 是否启用敏感信息检测
            max_workers: 最大工作线程数（暂未使用，保留兼容）
        """
        self.enable_audio = enable_audio
        self.enable_sensitive = enable_sensitive
        self.use_advanced_asr = use_advanced_asr

        # 构建分析器列表
        self.analyzers = []

        # 流畅度分析器（始终启用）
        from pipeline import SmoothnessAnalyzer
        self.analyzers.append(("smoothness", SmoothnessAnalyzer()))

        # 步态人脸分析器（始终启用）
        from pipeline import GaitFaceAnalyzerWrapper
        self.analyzers.append(("gait_face", GaitFaceAnalyzerWrapper()))

        # 音频分析器（可选）
        if enable_audio:
            from pipeline import AudioAnalyzerWrapper
            self.analyzers.append(("audio", AudioAnalyzerWrapper()))

        # 敏感检测器（可选）
        if enable_sensitive:
            from pipeline import SensitiveDetectorWrapper
            self.analyzers.append(("sensitive", SensitiveDetectorWrapper()))

    def analyze_video(self,
                      video_path: str,
                      task_id: str = "",
                      progress_callback: Optional[Callable] = None,
                      verbose: bool = True,
                      analyze_audio: bool = True,
                      detect_sensitive: bool = True) -> Dict:
        """
        分析视频文件

        Args:
            video_path: 视频文件路径
            task_id: 任务ID（可选）
            progress_callback: 进度回调函数
            verbose: 是否打印详细日志
            analyze_audio: 是否分析音频（覆盖初始化设置）
            detect_sensitive: 是否检测敏感信息（覆盖初始化设置）

        Returns:
            分析结果字典
        """

        def log(msg):
            if verbose:
                print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}")

        # 检查文件是否存在
        if not os.path.exists(video_path):
            return {"status": "error", "message": f"视频文件不存在: {video_path}"}

        # 1. 解码视频
        log("开始解码视频...")
        try:
            frames, audio_bytes = VideoDecoder.decode(video_path)
        except Exception as e:
            return {"status": "error", "message": f"解码失败: {str(e)}"}

        total_frames = len(frames)
        if total_frames == 0:
            return {"status": "error", "message": "视频无有效帧"}

        h, w = frames[0].shape[:2]
        fps = 30  # 默认值，实际应从视频信息获取
        duration = total_frames / fps

        log(f"视频信息：{total_frames} 帧, {w}x{h}, 时长: {duration:.2f}s")

        # 2. 构建 Pipeline
        log("构建分析管道...")
        pipeline = VideoPipeline()

        # 注册分析器
        from pipeline import SmoothnessAnalyzer, GaitFaceAnalyzerWrapper
        pipeline.register_analyzer(SmoothnessAnalyzer())
        pipeline.register_analyzer(GaitFaceAnalyzerWrapper())

        if analyze_audio and self.enable_audio:
            from pipeline import AudioAnalyzerWrapper
            pipeline.register_analyzer(AudioAnalyzerWrapper())
            if audio_bytes:
                pipeline.set_audio(audio_bytes)
                log(f"音频数据大小: {len(audio_bytes)} bytes")

        if detect_sensitive and self.enable_sensitive:
            from pipeline import SensitiveDetectorWrapper
            pipeline.register_analyzer(SensitiveDetectorWrapper())

        # 3. 设置进度回调
        if progress_callback:
            def on_progress(frame_idx):
                if frame_idx % 30 == 0:
                    progress = (frame_idx + 1) / total_frames * 100
                    progress_callback(progress)

            pipeline.on_frame_processed = on_progress

        # 4. 运行分析
        log("开始管道分析...")
        timestamps = [i / fps for i in range(total_frames)]

        try:
            results = pipeline.run(frames, timestamps)
        except Exception as e:
            return {"status": "error", "message": f"分析失败: {str(e)}"}

        # 5. 构建最终结果
        final_result = {
            "status": "success",
            "video_info": {
                "file_name": os.path.basename(video_path),
                "total_frames": total_frames,
                "fps": round(fps, 2),
                "duration": round(duration, 2),
                "width": w,
                "height": h
            },
            "analysis_results": results
        }

        log("✅ 分析完成！")
        return final_result

    @staticmethod
    def save_result_to_json(result: dict, output_dir: str = "results"):
        """保存结果到JSON文件"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            path = os.path.join(output_dir, filename)

            # 转换numpy类型
            serializable = tools.convert_to_serializable(result)

            with open(path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)

            print(f"✅ 结果已保存：{path}")
            return path
        except Exception as e:
            print(f"❌ 保存失败：{e}")
            return None


if __name__ == "__main__":
    core = VideoAnalysisCore()
    result = core.analyze_video("/Users/hutian/Downloads/anan.mov")
    core.save_result_to_json(result)