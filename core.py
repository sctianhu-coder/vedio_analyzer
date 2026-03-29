#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视频分析系统 - 核心模块
整合所有分析能力，提供统一调用接口
"""
import os
import time
import json
from datetime import datetime
from typing import Dict, Optional, Callable
import cv2
import numpy as np

# 导入工具函数
from tools import convert_to_serializable, ensure_dir, format_duration, convert_numpy_types

# 导入独立的分析模块
from video_smoothness import VideoSmoothnessAnalyzer
from gait_face_analyzer import GaitFaceAnalyzer
from audio_analyzer import AudioAnalyzer, AudioAnalyzerSimple
from sensitive_info_detector import SensitiveInfoDetector, PrivacyBlur


class VideoAnalysisCore:
    """视频分析核心类（整合所有分析能力）"""

    def __init__(self,
                 enable_audio: bool = True,
                 use_advanced_asr: bool = False,
                 enable_sensitive: bool = True,
                 ffmpeg_path: str = None):
        """
        初始化核心分析器
        :param enable_audio: 是否启用音频分析
        :param use_advanced_asr: 是否使用高级ASR（Whisper）
        :param enable_sensitive: 是否启用敏感信息检测
        :param ffmpeg_path: ffmpeg可执行文件路径
        """
        # 初始化各分析器
        self.smoothness_analyzer = VideoSmoothnessAnalyzer()
        self.gait_face_analyzer = GaitFaceAnalyzer()
        self.privacy_blur = PrivacyBlur() if enable_sensitive else None

        # 音频分析器初始化
        self.audio_analyzer = None
        if enable_audio:
            if use_advanced_asr:
                self.audio_analyzer = AudioAnalyzer(use_whisper=True)
            else:
                self.audio_analyzer = AudioAnalyzerSimple(ffmpeg_path=ffmpeg_path)

        # 敏感信息检测器
        self.sensitive_detector = SensitiveInfoDetector() if enable_sensitive else None

    def reset_analyzers(self):
        """重置所有分析器状态（复用实例时调用）"""
        self.smoothness_analyzer.reset()
        self.gait_face_analyzer.reset()

    def analyze_video(self,
                      video_path: str,
                      task_id: str = "",
                      progress_callback: Optional[Callable] = None,
                      verbose: bool = True,
                      analyze_audio: bool = True,
                      detect_sensitive: bool = True) -> Dict:
        """
        完整视频分析流程（兼容CLI/API两种调用方式）
        :param video_path: 视频文件路径
        :param task_id: 任务ID（API场景用）
        :param progress_callback: 进度回调函数
        :param verbose: 是否打印详细日志
        :param analyze_audio: 是否分析音频
        :param detect_sensitive: 是否检测敏感信息
        :return: 分析结果字典
        """
        # 重置分析器
        self.reset_analyzers()

        # 基础日志函数（兼容API/CLI）
        def log(msg: str):
            if verbose:
                prefix = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
                if task_id:
                    prefix += f" [任务:{task_id}]"
                print(f"{prefix} {msg}")

        # 1. 基础校验
        if not os.path.exists(video_path):
            log(f"错误：视频文件不存在 - {video_path}")
            return {"status": "error", "message": f"视频文件不存在: {video_path}"}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log(f"错误：无法打开视频 - {video_path}")
            return {"status": "error", "message": "无法打开视频"}

        # 2. 获取视频基础信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        # 修复苹果MOV帧尺寸突变问题（统一用第一帧尺寸）
        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            log("错误：视频为空")
            return {"status": "error", "message": "视频为空"}
        FIXED_WIDTH = first_frame.shape[1]
        FIXED_HEIGHT = first_frame.shape[0]
        log(f"✅ 统一帧尺寸：{FIXED_WIDTH}x{FIXED_HEIGHT}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 回到第0帧

        log(f"视频信息：总帧数={total_frames}，FPS={fps:.2f}，时长={duration:.1f}s")

        # 3. 逐帧分析
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = time.time()

            # 帧尺寸统一（修复突变问题）
            if frame.shape[1] != FIXED_WIDTH or frame.shape[0] != FIXED_HEIGHT:
                frame = cv2.resize(frame, (FIXED_WIDTH, FIXED_HEIGHT))

            # 流畅度分析
            self.smoothness_analyzer.evaluate_frame(frame, current_time)
            # 步态/人脸分析
            self.gait_face_analyzer.process_frame(frame)

            # 进度更新（API/CLI通用）
            if frame_count % 30 == 0:
                progress = min(round(frame_count / total_frames * 100, 1), 100)
                if progress_callback:
                    progress_callback(progress)
                log(f"处理进度：{progress}%")

        cap.release()

        # 4. 收集基础分析结果
        smooth_result = self.smoothness_analyzer.get_final_stats()
        gait_face_result = self.gait_face_analyzer.get_final_stats(total_frames, fps)
        audio_result = {}
        sensitive_result = {}

        # 5. 音频分析
        if analyze_audio and self.audio_analyzer:
            log("正在分析音频...")
            audio_result = self.audio_analyzer.analyze_video_audio(video_path)
            if audio_result.get("has_audio_content"):
                log(f"  音频：有声音 | 人声：{'检测到' if audio_result.get('has_voice') else '未检测到'}")

        # 6. 敏感信息检测
        if detect_sensitive and self.sensitive_detector:
            log("正在检测敏感信息...")
            transcript = audio_result.get('transcript', '')
            sensitive_result = self.sensitive_detector.analyze_video_sensitive(
                video_path, transcript=transcript, frame_interval=30
            )
            log(f"  敏感信息风险等级：{sensitive_result.get('overall_risk', 'unknown')}")

        # 7. 组装结果（兼容API/CLI输出格式）
        base_result = {
            "status": "success",
            "progress": 100,
            "timestamp": datetime.now().isoformat(),
            "video_info": {
                "file_path": os.path.abspath(video_path),
                "file_name": os.path.basename(video_path),
                "width": FIXED_WIDTH,
                "height": FIXED_HEIGHT,
                "fps": round(fps, 2),
                "total_frames": total_frames,
                "duration": round(duration, 2),
                "duration_formatted": format_duration(duration)
            },
            "smoothness": convert_numpy_types(smooth_result),
            "face_detection": convert_numpy_types(gait_face_result["face_detection"]),
            "gait_analysis": convert_numpy_types(gait_face_result["gait_analysis"]),
            "audio": convert_numpy_types(audio_result),
            "sensitive_detection": convert_numpy_types(sensitive_result)
        }

        # 8. 生成总结（CLI场景用）
        base_result["summary"] = self._generate_summary(
            smooth_result, gait_face_result["gait_analysis"],
            gait_face_result["face_detection"], audio_result, sensitive_result
        )

        log("✅ 全部分析完成")
        return base_result

    def _generate_summary(self, smooth_stats: Dict, gait_stats: Dict,
                          face_stats: Dict, audio_result: Dict, sensitive_result: Dict) -> str:
        """生成分析总结文本"""
        summary = []
        # 流畅度
        score = smooth_stats['overall_score']
        summary.append(f"视频流畅度{'良好' if score >= 75 else '待提升'}({score:.0f}分)")
        # 步态
        if gait_stats['step_count'] > 0:
            summary.append(f"检测到{gait_stats['step_count']}步，步频{gait_stats['cadence']:.0f}步/分钟")
            summary.append(f"左右腿步态{'对称' if gait_stats['knee_angles']['symmetry'] >= 80 else '不对称'}")
        # 人脸
        summary.append(
            f"{'检测到人脸（覆盖{face_stats["face_coverage"]}%帧）' if face_stats['has_face'] else '未检测到人脸'}")
        # 音频
        if audio_result.get("has_audio_content"):
            summary.append(f"{'检测到人声' if audio_result.get('has_voice') else '有背景音但无人声'}")
        else:
            summary.append("视频无声")
        # 敏感信息
        if sensitive_result:
            risk = sensitive_result.get('overall_risk', 'low')
            summary.append(f"内容风险：{risk}（{'建议审核' if risk in ['high', 'medium'] else '安全'}）")
        return "；".join(summary)

    # 工具方法（复用原app.py逻辑）
    @staticmethod
    def save_result_to_json(result: Dict, output_dir: str = "output", output_file: str = None) -> str:
        """保存结果到JSON文件"""
        ensure_dir(output_dir)
        if not output_file:
            video_name = result['video_info']['file_name']
            base_name = os.path.splitext(video_name)[0]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"{base_name}_{timestamp}.json"
        json_file = os.path.join(output_dir, output_file)

        serializable_result = convert_to_serializable(result)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)
        return json_file

    @staticmethod
    def print_json_result(result: Dict):
        """打印格式化的JSON结果"""
        serializable_result = convert_to_serializable(result)
        print(json.dumps(serializable_result, ensure_ascii=False, indent=2))