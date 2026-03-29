#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
管道式视频分析框架
每个分析器独立处理，数据流过各个模块，最后汇总结果

使用示例:
    from pipeline import PipelineBuilder, VideoPipeline

    # 方式1: 使用构建器
    pipeline = PipelineBuilder.build_full_pipeline()
    pipeline.set_audio(audio_bytes)
    results = pipeline.run(frames, timestamps)

    # 方式2: 自定义管道
    pipeline = VideoPipeline()
    pipeline.register_analyzer(SmoothnessAnalyzer())
    pipeline.register_analyzer(GaitFaceAnalyzerWrapper())
    results = pipeline.run(frames)
"""

import os
import time
import threading
import queue
import numpy as np
from typing import Dict, Optional, Callable, List, Any
from dataclasses import dataclass
import uuid


# ============================================================
# 数据结构
# ============================================================

@dataclass
class FrameData:
    """单帧数据结构"""
    frame: np.ndarray
    timestamp: float
    frame_index: int
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AudioData:
    """音频数据结构"""
    audio_bytes: bytes
    sample_rate: int = 16000
    duration: float = 0.0
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AnalysisResult:
    """单个分析器的结果"""

    def __init__(self, analyzer_name: str):
        self.analyzer_name = analyzer_name
        self.data: Dict = {}
        self.status: str = "pending"
        self.error: Optional[str] = None
        self.timestamp = time.time()

    def to_dict(self) -> Dict:
        return {
            "analyzer": self.analyzer_name,
            "data": self.data,
            "status": self.status,
            "error": self.error,
            "timestamp": self.timestamp
        }


# ============================================================
# 分析器基类
# ============================================================

class BaseAnalyzer:
    """分析器基类 - 所有分析器都要继承此类"""

    def __init__(self, name: str):
        self.name = name
        self.result = AnalysisResult(name)

    def process_frame(self, frame_data: FrameData) -> Optional[Dict]:
        """处理单帧（子类实现）"""
        raise NotImplementedError

    def process_audio(self, audio_data: AudioData) -> Optional[Dict]:
        """处理音频（子类可选实现）"""
        return None

    def get_result(self) -> Dict:
        """获取分析结果"""
        return self.result.to_dict()

    def reset(self):
        """重置分析器状态"""
        self.result = AnalysisResult(self.name)


# ============================================================
# 具体分析器实现
# ============================================================

class SmoothnessAnalyzer(BaseAnalyzer):
    """流畅度分析器"""

    def __init__(self):
        super().__init__("smoothness")
        from video_smoothness import VideoSmoothnessAnalyzer
        self._analyzer = VideoSmoothnessAnalyzer()
        self.smoothness_scores = []
        self.stutter_count = 0
        self.freeze_count = 0

    def process_frame(self, frame_data: FrameData) -> Optional[Dict]:
        result = self._analyzer.evaluate_frame(frame_data.frame, frame_data.timestamp)

        self.smoothness_scores.append(result.get('score', 0))
        if result.get('is_stutter'):
            self.stutter_count += 1
        if result.get('is_freeze'):
            self.freeze_count += 1

        return result

    def get_result(self) -> Dict:
        avg_score = np.mean(self.smoothness_scores) if self.smoothness_scores else 0

        self.result.data = {
            "overall_score": round(avg_score, 1),
            "score_level": self._get_level(avg_score),
            "stutter_count": self.stutter_count,
            "freeze_count": self.freeze_count,
            "total_frames_analyzed": len(self.smoothness_scores),
            "is_smooth": avg_score >= 75
        }
        self.result.status = "completed"
        return self.result.to_dict()

    @staticmethod
    def _get_level(score: float) -> str:
        if score >= 90:
            return "极佳"
        elif score >= 75:
            return "良好"
        elif score >= 60:
            return "一般"
        elif score >= 40:
            return "较差"
        else:
            return "很差"

    def reset(self):
        super().reset()
        from video_smoothness import VideoSmoothnessAnalyzer
        self._analyzer = VideoSmoothnessAnalyzer()
        self.smoothness_scores = []
        self.stutter_count = 0
        self.freeze_count = 0


class GaitFaceAnalyzerWrapper(BaseAnalyzer):
    """步态人脸分析器包装器"""

    def __init__(self):
        super().__init__("gait_face")
        from gait_face_analyzer import GaitFaceAnalyzer
        self._analyzer = GaitFaceAnalyzer(show_window=False)
        self.frame_results = []
        self.total_frames = 0

    def process_frame(self, frame_data: FrameData) -> Optional[Dict]:
        try:
            result = self._analyzer.process_frame(frame_data.frame)
            self.frame_results.append(result)
            self.total_frames += 1
            return result
        except Exception as e:
            print(f"GaitFaceAnalyzer 处理帧错误: {e}")
            return None

    def get_result(self) -> Dict:
        fps = 30
        try:
            final_stats = self._analyzer.get_final_stats(self.total_frames, fps)
            self.result.data = {
                "gait_analysis": final_stats["gait_analysis"],
                "face_detection": final_stats["face_detection"]
            }
        except Exception as e:
            self.result.data = {
                "gait_analysis": {"error": str(e)},
                "face_detection": {"error": str(e)}
            }
        self.result.status = "completed"
        return self.result.to_dict()

    def reset(self):
        super().reset()
        from gait_face_analyzer import GaitFaceAnalyzer
        self._analyzer = GaitFaceAnalyzer(show_window=False)
        self.frame_results = []
        self.total_frames = 0


class AudioAnalyzerWrapper(BaseAnalyzer):
    """音频分析器包装器"""

    def __init__(self):
        super().__init__("audio")
        from audio_analyzer import AudioAnalyzerSimple
        self._analyzer = AudioAnalyzerSimple()
        self._processed = False

    def process_audio(self, audio_data: AudioData) -> Optional[Dict]:
        if self._processed:
            return self.result.data

        result = self._analyzer.analyze_audio_from_bytes(audio_data.audio_bytes)
        self.result.data = result
        self.result.status = "completed"
        self._processed = True
        return result

    def get_result(self) -> Dict:
        if self.result.status == "pending":
            return {"analyzer": self.name, "status": "pending", "data": {}}
        return self.result.to_dict()

    def reset(self):
        super().reset()
        self._processed = False


class SensitiveDetectorWrapper(BaseAnalyzer):
    """敏感信息检测器包装器"""

    def __init__(self, frame_interval: int = 30):
        super().__init__("sensitive")
        from sensitive_info_detector import SensitiveInfoDetector
        self._detector = SensitiveInfoDetector()
        self.frame_interval = frame_interval
        self.frame_samples = []
        self.transcript = ""

    def process_frame(self, frame_data: FrameData) -> Optional[Dict]:
        # 采样分析
        if frame_data.frame_index % self.frame_interval == 0:
            result = self._detector.analyze_frame(frame_data.frame)
            self.frame_samples.append(result)
        return None

    def set_transcript(self, transcript: str):
        self.transcript = transcript

    def get_result(self) -> Dict:
        if not self.frame_samples:
            self.result.data = {"status": "no_frames_analyzed"}
            self.result.status = "completed"
            return self.result.to_dict()

        max_faces = max([s.get('face_privacy', {}).get('face_count', 0) for s in self.frame_samples])
        max_skin = max([s.get('skin_content', {}).get('skin_ratio', 0) for s in self.frame_samples])
        high_risk = sum(1 for s in self.frame_samples if s.get('overall_risk') == 'high')

        self.result.data = {
            "has_faces": max_faces > 0,
            "max_faces_per_frame": max_faces,
            "max_skin_ratio": round(max_skin, 2),
            "high_risk_frames": high_risk,
            "risk_ratio": round(high_risk / len(self.frame_samples) * 100, 2),
            "transcript_analyzed": bool(self.transcript),
            "overall_risk": "high" if high_risk > len(self.frame_samples) * 0.3
            else "medium" if high_risk > 0 else "low"
        }
        self.result.status = "completed"
        return self.result.to_dict()

    def reset(self):
        super().reset()
        self.frame_samples = []
        self.transcript = ""


# ============================================================
# 视频管道
# ============================================================

class VideoPipeline:
    """
    视频处理管道
    数据像流水线一样流过各个分析器
    """

    def __init__(self, max_queue_size: int = 30):
        self.max_queue_size = max_queue_size

        # 注册的分析器
        self.analyzers: List[BaseAnalyzer] = []

        # 帧队列
        self.frame_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.audio_data: Optional[AudioData] = None

        # 控制标志
        self.stop_event = threading.Event()
        self.is_running = False

        # 结果收集
        self.results: Dict[str, Dict] = {}

        # 回调
        self.on_frame_processed: Optional[Callable[[int], None]] = None
        self.on_complete: Optional[Callable[[Dict], None]] = None

    def register_analyzer(self, analyzer: BaseAnalyzer):
        """注册分析器"""
        self.analyzers.append(analyzer)
        print(f"📎 已注册分析器: {analyzer.name}")

    def register_analyzers(self, analyzers: List[BaseAnalyzer]):
        """批量注册分析器"""
        for analyzer in analyzers:
            self.register_analyzer(analyzer)

    def unregister_analyzer(self, analyzer_name: str):
        """注销分析器"""
        self.analyzers = [a for a in self.analyzers if a.name != analyzer_name]

    def set_audio(self, audio_bytes: bytes, sample_rate: int = 16000):
        """设置音频数据"""
        self.audio_data = AudioData(audio_bytes, sample_rate)

    def set_transcript(self, transcript: str):
        """设置语音转录文本（用于敏感检测）"""
        for analyzer in self.analyzers:
            if hasattr(analyzer, 'set_transcript'):
                analyzer.set_transcript(transcript)

    def _process_frame_worker(self):
        """帧处理工作线程"""
        while not self.stop_event.is_set():
            try:
                frame_data = self.frame_queue.get(timeout=0.5)
                if frame_data is None:  # 结束信号
                    break

                # 让每个分析器处理这一帧
                for analyzer in self.analyzers:
                    if hasattr(analyzer, 'process_frame'):
                        analyzer.process_frame(frame_data)

                # 进度回调
                if self.on_frame_processed:
                    self.on_frame_processed(frame_data.frame_index)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"帧处理错误: {e}")

    def _process_audio(self):
        """处理音频"""
        if self.audio_data:
            for analyzer in self.analyzers:
                if hasattr(analyzer, 'process_audio'):
                    analyzer.process_audio(self.audio_data)

    def run(self, frames: np.ndarray, timestamps: List[float] = None) -> Dict:
        """
        运行管道处理

        Args:
            frames: 视频帧数组 (n_frames, height, width, 3)
            timestamps: 每帧的时间戳（可选，默认按30fps计算）

        Returns:
            分析结果字典
        """
        self.is_running = True
        self.stop_event.clear()

        total_frames = len(frames)

        # 生成时间戳
        if timestamps is None:
            fps = 30
            timestamps = [i / fps for i in range(total_frames)]

        # 启动工作线程
        worker_thread = threading.Thread(target=self._process_frame_worker, daemon=True)
        worker_thread.start()

        # 推送帧到队列
        print(f"📤 开始推送 {total_frames} 帧到管道...")
        for i, (frame, ts) in enumerate(zip(frames, timestamps)):
            if self.stop_event.is_set():
                break
            frame_data = FrameData(frame, ts, i)
            self.frame_queue.put(frame_data)

        # 发送结束信号
        self.frame_queue.put(None)
        worker_thread.join(timeout=5)

        # 处理音频
        self._process_audio()

        # 收集结果
        self._collect_results()

        self.is_running = False
        print(f"✅ 管道处理完成，共处理 {total_frames} 帧")

        if self.on_complete:
            self.on_complete(self.results)

        return self.results

    def _collect_results(self):
        """收集所有分析器的结果"""
        self.results = {}
        for analyzer in self.analyzers:
            result = analyzer.get_result()
            self.results[analyzer.name] = result

    def get_result(self) -> Dict:
        """获取最终结果"""
        return self.results

    def stop(self):
        """停止处理"""
        self.stop_event.set()

    def reset(self):
        """重置所有分析器"""
        for analyzer in self.analyzers:
            analyzer.reset()
        self.results = {}
        # 重新创建队列
        self.frame_queue = queue.Queue(maxsize=self.max_queue_size)


# ============================================================
# 管道构建器
# ============================================================

class PipelineBuilder:
    """管道构建器 - 用于快速创建常用管道配置"""

    @staticmethod
    def build_full_pipeline() -> VideoPipeline:
        """
        构建完整分析管道
        包含：流畅度、步态人脸、音频、敏感检测
        """
        pipeline = VideoPipeline()
        pipeline.register_analyzer(SmoothnessAnalyzer())
        pipeline.register_analyzer(GaitFaceAnalyzerWrapper())
        pipeline.register_analyzer(AudioAnalyzerWrapper())
        pipeline.register_analyzer(SensitiveDetectorWrapper())
        return pipeline

    @staticmethod
    def build_light_pipeline() -> VideoPipeline:
        """
        构建轻量级管道
        只包含：流畅度、步态人脸
        """
        pipeline = VideoPipeline()
        pipeline.register_analyzer(SmoothnessAnalyzer())
        pipeline.register_analyzer(GaitFaceAnalyzerWrapper())
        return pipeline

    @staticmethod
    def build_smoothness_only() -> VideoPipeline:
        """只分析流畅度"""
        pipeline = VideoPipeline()
        pipeline.register_analyzer(SmoothnessAnalyzer())
        return pipeline

    @staticmethod
    def build_gait_face_only() -> VideoPipeline:
        """只分析步态和人脸"""
        pipeline = VideoPipeline()
        pipeline.register_analyzer(GaitFaceAnalyzerWrapper())
        return pipeline

    @staticmethod
    def build_audio_only() -> VideoPipeline:
        """只分析音频"""
        pipeline = VideoPipeline()
        pipeline.register_analyzer(AudioAnalyzerWrapper())
        return pipeline

    @staticmethod
    def build_custom(analyzers: List[str]) -> VideoPipeline:
        """
        根据名称列表构建自定义管道

        Args:
            analyzers: 分析器名称列表，可选: smoothness, gait_face, audio, sensitive

        Returns:
            VideoPipeline 实例
        """
        analyzer_map = {
            "smoothness": SmoothnessAnalyzer,
            "gait_face": GaitFaceAnalyzerWrapper,
            "audio": AudioAnalyzerWrapper,
            "sensitive": SensitiveDetectorWrapper,
        }

        pipeline = VideoPipeline()
        for name in analyzers:
            if name in analyzer_map:
                pipeline.register_analyzer(analyzer_map[name]())
            else:
                print(f"⚠️ 未知分析器: {name}")

        return pipeline


# ============================================================
# 便捷函数
# ============================================================

def analyze_with_pipeline(frames: np.ndarray,
                          audio_bytes: bytes = None,
                          analyzers: List[str] = None) -> Dict:
    """
    使用管道分析视频的便捷函数

    Args:
        frames: 视频帧数组
        audio_bytes: 音频数据（可选）
        analyzers: 分析器列表，默认使用全部

    Returns:
        分析结果
    """
    if analyzers is None:
        analyzers = ["smoothness", "gait_face", "audio", "sensitive"]

    pipeline = PipelineBuilder.build_custom(analyzers)

    if audio_bytes:
        pipeline.set_audio(audio_bytes)

    timestamps = [i / 30 for i in range(len(frames))]
    return pipeline.run(frames, timestamps)


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("管道框架测试")
    print("=" * 60)

    # 创建模拟数据
    dummy_frames = np.random.randint(0, 255, (100, 480, 640, 3), dtype=np.uint8)
    dummy_audio = b"\x00" * 16000 * 2  # 1秒静音

    print(f"模拟数据: {len(dummy_frames)} 帧, 音频 {len(dummy_audio)} bytes")

    # 测试完整管道
    print("\n1. 测试完整管道...")
    pipeline = PipelineBuilder.build_full_pipeline()
    pipeline.set_audio(dummy_audio)


    def on_progress(idx):
        if idx % 20 == 0:
            print(f"   进度: {idx}/{len(dummy_frames)}")


    pipeline.on_frame_processed = on_progress
    results = pipeline.run(dummy_frames)

    print("\n2. 测试轻量级管道...")
    light_pipeline = PipelineBuilder.build_light_pipeline()
    light_results = light_pipeline.run(dummy_frames)

    print("\n✅ 测试完成")