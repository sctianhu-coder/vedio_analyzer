"""
视频流畅度分析模块
独立的流畅度分析类，不依赖 MediaPipe
"""

import cv2
import numpy as np
from collections import deque
import time
from typing import Dict, Optional, Tuple


class VideoSmoothnessAnalyzer:
    """
    视频流畅度分析器

    评估指标：
    1. FPS稳定性（30%权重）：帧率波动程度，标准差越小越流畅
    2. 运动平滑度（30%权重）：使用光流法检测运动连续性
    3. 卡顿检测（20%权重）：检测画面突变或停顿
    4. 画面冻结（20%权重）：检测完全静止的帧
    """

    def __init__(self):
        """初始化流畅度分析器"""
        # 帧间差异历史
        self.frame_diffs = deque(maxlen=30)  # 帧间差异值
        self.motion_vectors = deque(maxlen=30)  # 运动向量
        self.fps_history = deque(maxlen=60)  # FPS历史

        # 分析结果
        self.smoothness_score = 0.0
        self.stutter_count = 0
        self.freeze_count = 0

        # 上一帧数据
        self.prev_frame = None
        self.prev_gray = None
        self.prev_time = None

        # 是否已重置
        self.is_reset = True

    def reset(self):
        """重置所有状态"""
        self.frame_diffs.clear()
        self.motion_vectors.clear()
        self.fps_history.clear()
        self.smoothness_score = 0.0
        self.stutter_count = 0
        self.freeze_count = 0
        self.prev_frame = None
        self.prev_gray = None
        self.prev_time = None
        self.is_reset = True

    def calculate_frame_difference(self, frame: np.ndarray) -> float:
        """
        计算帧间差异（检测卡顿）

        Args:
            frame: 当前帧图像

        Returns:
            帧间差异值
        """
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            return 0

        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)

        # 计算绝对差异
        diff = cv2.absdiff(gray, prev_gray)

        # 计算差异的均值
        mean_diff = np.mean(diff)

        # 计算变化像素比例
        changed_pixels = np.sum(diff > 30) / diff.size

        # 综合差异值
        frame_diff = mean_diff * (1 + changed_pixels)

        self.frame_diffs.append(frame_diff)
        self.prev_frame = frame.copy()

        return frame_diff

    def detect_stutter(self, frame_diff: float) -> bool:
        """
        检测卡顿（画面突变）

        Args:
            frame_diff: 当前帧差异值

        Returns:
            是否检测到卡顿
        """
        if len(self.frame_diffs) < 5:
            return False

        # 计算平均差异
        avg_diff = np.mean(list(self.frame_diffs)[-5:])

        # 如果当前差异远小于平均差异，可能是卡顿
        if frame_diff < avg_diff * 0.3 and avg_diff > 10:
            self.stutter_count += 1
            return True

        return False

    def detect_freeze(self, frame: np.ndarray) -> bool:
        """
        检测画面冻结

        Args:
            frame: 当前帧图像

        Returns:
            是否检测到冻结
        """
        if self.prev_gray is None:
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 计算帧间差异
        diff = cv2.absdiff(gray, self.prev_gray)
        mean_diff = np.mean(diff)

        # 如果差异很小，可能是冻结
        if mean_diff < 1.0:
            self.freeze_count += 1
            self.prev_gray = gray
            return True

        self.prev_gray = gray
        return False

    def calculate_motion_smoothness(self, frame: np.ndarray) -> float:
        """
        计算运动平滑度（使用光流法）

        Args:
            frame: 当前帧图像

        Returns:
            运动平滑度 (0-1)
        """
        if self.prev_gray is None:
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return 1.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 计算光流
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        # 计算运动向量的大小
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_mag = np.mean(mag)

        # 计算运动幅度变化率
        if len(self.motion_vectors) > 0:
            prev_mag = self.motion_vectors[-1]
            smoothness = 1.0 - min(1.0, abs(mean_mag - prev_mag) / (prev_mag + 1))
        else:
            smoothness = 1.0

        self.motion_vectors.append(mean_mag)
        self.prev_gray = gray

        return smoothness

    def calculate_fps_stability(self, current_time: float) -> float:
        """
        计算FPS稳定性

        Args:
            current_time: 当前时间戳

        Returns:
            FPS稳定性评分 (0-100)
        """
        if self.prev_time is None:
            self.prev_time = current_time
            return 100.0

        # 计算当前帧间隔
        frame_interval = (current_time - self.prev_time) * 1000  # 转换为毫秒
        self.prev_time = current_time

        # 记录FPS
        current_fps = 1000.0 / frame_interval if frame_interval > 0 else 0
        self.fps_history.append(current_fps)

        if len(self.fps_history) < 10:
            return 100.0

        # 计算FPS标准差（越小越稳定）
        fps_std = np.std(list(self.fps_history))

        # FPS稳定性评分（0-100）
        fps_stability = max(0, 100 - fps_std * 2)

        return fps_stability

    def evaluate_frame(self, frame: np.ndarray, timestamp: float = None) -> Dict:
        """
        评估单帧的流畅度

        Args:
            frame: 当前帧图像
            timestamp: 时间戳，默认使用当前时间

        Returns:
            当前帧的流畅度指标
        """
        if timestamp is None:
            timestamp = time.time()

        # 计算各项指标
        frame_diff = self.calculate_frame_difference(frame)
        is_stutter = self.detect_stutter(frame_diff)
        is_freeze = self.detect_freeze(frame)
        motion_smooth = self.calculate_motion_smoothness(frame)
        fps_stability = self.calculate_fps_stability(timestamp)

        # 综合评分（0-100）
        stability_score = fps_stability * 0.3
        motion_score = motion_smooth * 30
        stutter_score = max(0, 100 - self.stutter_count * 5) * 0.2
        freeze_score = max(0, 100 - self.freeze_count * 2) * 0.2

        self.smoothness_score = stability_score + motion_score + stutter_score + freeze_score
        self.smoothness_score = min(100, max(0, self.smoothness_score))

        return {
            'score': self.smoothness_score,
            'fps_stability': round(fps_stability, 1),
            'motion_smoothness': round(motion_smooth * 100, 1),
            'stutter_count': self.stutter_count,
            'freeze_count': self.freeze_count,
            'frame_diff': round(frame_diff, 2),
            'is_stutter': is_stutter,
            'is_freeze': is_freeze
        }

    def get_final_stats(self) -> Dict:
        """
        获取最终统计结果

        Returns:
            最终的流畅度统计结果
        """
        # 计算平均运动平滑度
        avg_motion_smooth = np.mean(self.motion_vectors) if self.motion_vectors else 0

        # 计算平均FPS稳定性
        avg_fps_stability = np.mean(self.fps_history) if self.fps_history else 0
        fps_std = np.std(list(self.fps_history)) if self.fps_history else 0

        return {
            "overall_score": round(self.smoothness_score, 1),
            "score_level": self._get_smoothness_level(self.smoothness_score),
            "motion_smoothness": round(avg_motion_smooth * 100, 1),
            "avg_fps": round(avg_fps_stability, 1),
            "fps_std": round(fps_std, 2),
            "stutter_frames": self.stutter_count,
            "freeze_frames": self.freeze_count,
            "is_smooth": self.smoothness_score >= 75
        }

    @staticmethod
    def _get_smoothness_level(score: float) -> str:
        """获取流畅度等级"""
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