"""
步态与人脸识别模块
独立的步态分析和人脸检测类
"""

import cv2
import mediapipe as mp
import numpy as np
import threading
from typing import Dict, Optional, List, Tuple


class GaitFaceAnalyzer:
    """
    步态与人脸分析器 - 线程安全版本
    每个实例独立，使用线程本地存储
    """

    def __init__(self, show_window: bool = False):
        """初始化分析器"""
        self.mp_pose = mp.solutions.pose
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils

        # 控制是否弹出窗口
        self.show_window = show_window

        # 线程本地存储 - 每个线程有自己的 MediaPipe 实例
        self._local = threading.local()

        # 步态数据
        self.gait_data = {
            'left_knee_angles': [],
            'right_knee_angles': [],
            'step_count': 0,
            'total_frames_processed': 0
        }

        # 人脸数据
        self.face_data = {
            'face_detected_frames': 0,
            'total_faces_detected': 0,
            'max_faces_in_frame': 0
        }

        # 临时变量
        self.prev_left_angle = None
        self.frame_count = 0

        # 线程锁保护共享数据
        self._data_lock = threading.Lock()

    def _get_pose(self):
        """获取当前线程的 Pose 实例"""
        if not hasattr(self._local, 'pose'):
            self._local.pose = self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        return self._local.pose

    def _get_face_detection(self):
        """获取当前线程的 FaceDetection 实例"""
        if not hasattr(self._local, 'face_detection'):
            self._local.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5
            )
        return self._local.face_detection

    def reset(self):
        """重置所有数据"""
        with self._data_lock:
            self.gait_data = {
                'left_knee_angles': [],
                'right_knee_angles': [],
                'step_count': 0,
                'total_frames_processed': 0
            }
            self.face_data = {
                'face_detected_frames': 0,
                'total_faces_detected': 0,
                'max_faces_in_frame': 0
            }
            self.prev_left_angle = None
            self.frame_count = 0

    def get_current_state(self) -> Dict:
        """获取当前分析器状态"""
        with self._data_lock:
            return {
                'gait_data': {
                    'left_knee_angles': self.gait_data['left_knee_angles'].copy(),
                    'right_knee_angles': self.gait_data['right_knee_angles'].copy(),
                    'step_count': self.gait_data['step_count'],
                    'total_frames_processed': self.gait_data['total_frames_processed']
                },
                'face_data': {
                    'face_detected_frames': self.face_data['face_detected_frames'],
                    'total_faces_detected': self.face_data['total_faces_detected'],
                    'max_faces_in_frame': self.face_data['max_faces_in_frame']
                },
                'prev_left_angle': self.prev_left_angle,
                'frame_count': self.frame_count,
                'show_window': self.show_window
            }

    def merge_state(self, state: Dict):
        """合并外部状态"""
        with self._data_lock:
            if 'gait_data' in state:
                self.gait_data['left_knee_angles'].extend(state['gait_data']['left_knee_angles'])
                self.gait_data['right_knee_angles'].extend(state['gait_data']['right_knee_angles'])
                self.gait_data['step_count'] = state['gait_data']['step_count']
                self.gait_data['total_frames_processed'] = state['gait_data']['total_frames_processed']

            if 'face_data' in state:
                self.face_data['face_detected_frames'] = state['face_data']['face_detected_frames']
                self.face_data['total_faces_detected'] = state['face_data']['total_faces_detected']
                self.face_data['max_faces_in_frame'] = max(
                    self.face_data['max_faces_in_frame'],
                    state['face_data']['max_faces_in_frame']
                )

            if 'prev_left_angle' in state:
                self.prev_left_angle = state['prev_left_angle']
            if 'frame_count' in state:
                self.frame_count = state['frame_count']
            if 'show_window' in state:
                self.show_window = state['show_window']

    def process_frame(self, frame: np.ndarray) -> Dict:
        """处理单帧图像"""
        # 更新帧计数
        with self._data_lock:
            self.frame_count += 1
            current_frame = self.frame_count

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_result = {
            'has_face': False,
            'faces_count': 0,
            'has_pose': False,
            'left_knee': None,
            'right_knee': None
        }

        # ========== 人脸检测（使用线程本地实例）==========
        try:
            face_detection = self._get_face_detection()
            face_results = face_detection.process(rgb_frame)

            if face_results.detections:
                faces_count = len(face_results.detections)
                with self._data_lock:
                    self.face_data['face_detected_frames'] += 1
                    self.face_data['total_faces_detected'] += faces_count
                    self.face_data['max_faces_in_frame'] = max(
                        self.face_data['max_faces_in_frame'],
                        faces_count
                    )

                frame_result['has_face'] = True
                frame_result['faces_count'] = faces_count
        except Exception as e:
            # 忽略 MediaPipe 的警告信息
            pass

        # ========== 姿态检测（使用线程本地实例）==========
        try:
            pose = self._get_pose()
            pose_results = pose.process(rgb_frame)

            if pose_results.pose_landmarks:
                with self._data_lock:
                    self.gait_data['total_frames_processed'] += 1

                frame_result['has_pose'] = True

                landmarks = pose_results.pose_landmarks.landmark

                # 计算角度
                left_knee = self._calc_knee_angle(landmarks, 'left')
                right_knee = self._calc_knee_angle(landmarks, 'right')

                with self._data_lock:
                    if left_knee:
                        self.gait_data['left_knee_angles'].append(left_knee)
                        frame_result['left_knee'] = left_knee
                    if right_knee:
                        self.gait_data['right_knee_angles'].append(right_knee)
                        frame_result['right_knee'] = right_knee

                    # 步数计算
                    if self.prev_left_angle and left_knee:
                        if abs(left_knee - self.prev_left_angle) > 20:
                            self.gait_data['step_count'] += 1
                    self.prev_left_angle = left_knee
        except Exception as e:
            # 忽略 MediaPipe 的警告信息
            pass

        return frame_result

    def get_final_stats(self, total_frames: int, fps: float) -> Dict:
        """获取最终结果"""
        with self._data_lock:
            if self.gait_data['total_frames_processed'] > 0 and fps > 0:
                duration_processed = self.gait_data['total_frames_processed'] / fps
                cadence = (self.gait_data['step_count'] / duration_processed) * 60 if duration_processed > 0 else 0
            else:
                cadence = 0

            avg_left = np.mean(self.gait_data['left_knee_angles']) if self.gait_data['left_knee_angles'] else 0
            avg_right = np.mean(self.gait_data['right_knee_angles']) if self.gait_data['right_knee_angles'] else 0

            symmetry = 100 - abs(avg_left - avg_right) / 90 * 100 if (avg_left + avg_right) > 0 else 0
            has_face = self.face_data['face_detected_frames'] > 0
            face_coverage = (self.face_data['face_detected_frames'] / total_frames * 100) if total_frames > 0 else 0

            return {
                "gait_analysis": {
                    "step_count": self.gait_data['step_count'],
                    "cadence": round(cadence, 1),
                    "cadence_level": self._get_cadence_level(cadence),
                    "knee_angles": {
                        "left": round(avg_left, 1),
                        "right": round(avg_right, 1),
                        "symmetry": round(symmetry, 1),
                        "symmetry_status": self._get_symmetry_status(symmetry)
                    },
                    "detection_rate": round(self.gait_data['total_frames_processed'] / total_frames * 100, 1) if total_frames > 0 else 0
                },
                "face_detection": {
                    "has_face": has_face,
                    "face_coverage": round(face_coverage, 1),
                    "face_detected_frames": self.face_data['face_detected_frames'],
                    "total_faces_detected": self.face_data['total_faces_detected'],
                    "max_faces_in_frame": self.face_data['max_faces_in_frame'],
                    "avg_faces_per_frame": round(self.face_data['total_faces_detected'] / total_frames, 2) if total_frames > 0 else 0,
                    "description": self._get_face_description(has_face, face_coverage, self.face_data['max_faces_in_frame'])
                }
            }

    def _calc_knee_angle(self, landmarks, side: str) -> Optional[float]:
        """计算膝关节角度"""
        try:
            if side == 'left':
                hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
                knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
                ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
            else:
                hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
                knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
                ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

            return self._angle_3d([hip.x, hip.y], [knee.x, knee.y], [ankle.x, ankle.y])
        except Exception:
            return None

    def _angle_3d(self, a, b, c) -> float:
        """计算三点角度"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    @staticmethod
    def _get_cadence_level(cadence: float) -> str:
        if cadence < 80:
            return "慢走"
        elif cadence < 110:
            return "正常行走"
        elif cadence < 140:
            return "慢跑"
        else:
            return "快速跑步"

    @staticmethod
    def _get_symmetry_status(symmetry: float) -> str:
        if symmetry >= 80:
            return "良好"
        elif symmetry >= 60:
            return "一般"
        else:
            return "较差"

    @staticmethod
    def _get_face_description(has_face: bool, face_coverage: float, max_faces: int) -> str:
        if not has_face:
            return "未检测到人脸"
        if max_faces == 1:
            return f"单人视频，人脸覆盖 {face_coverage}% 的帧"
        else:
            return f"多人视频（最多{max_faces}人），人脸覆盖 {face_coverage}% 的帧"