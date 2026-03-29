"""
步态与人脸识别模块
独立的步态分析和人脸检测类
带可视化窗口弹出（人脸框 + 骨骼点）
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, Optional, List, Tuple


class GaitFaceAnalyzer:
    """
    步态与人脸分析器
    功能：
    1. 步态分析：步数、步频、膝关节角度、左右对称性
    2. 人脸检测：是否有人脸、人脸覆盖率、人脸数量统计
    3. 可视化窗口：实时显示人脸框 + 姿态骨骼
    """

    def __init__(self, show_window: bool = True):
        """初始化分析器
        :param show_window: 是否弹出可视化窗口（默认开启）
        """
        self.mp_pose = mp.solutions.pose
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils  # 用于绘制

        # 控制是否弹出窗口
        self.show_window = show_window

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

    def reset(self):
        """重置所有数据"""
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

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        处理单帧图像，检测步态和人脸 + 绘制窗口
        """
        self.frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 用于绘制的副本
        display_frame = frame.copy()

        frame_result = {
            'has_face': False,
            'faces_count': 0,
            'has_pose': False,
            'left_knee': None,
            'right_knee': None
        }

        # ========== 人脸检测 ==========
        with self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            face_results = face_detection.process(rgb_frame)

            if face_results.detections:
                faces_count = len(face_results.detections)
                self.face_data['face_detected_frames'] += 1
                self.face_data['total_faces_detected'] += faces_count
                self.face_data['max_faces_in_frame'] = max(self.face_data['max_faces_in_frame'], faces_count)

                frame_result['has_face'] = True
                frame_result['faces_count'] = faces_count

                # ======================
                # 画出人脸框（关键！）
                # ======================
                for detection in face_results.detections:
                    self.mp_drawing.draw_detection(display_frame, detection)

        # ========== 姿态检测 ==========
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            pose_results = pose.process(rgb_frame)

            if pose_results.pose_landmarks:
                self.gait_data['total_frames_processed'] += 1
                frame_result['has_pose'] = True

                landmarks = pose_results.pose_landmarks

                # ======================
                # 画出姿态骨骼（关键！）
                # ======================
                self.mp_drawing.draw_landmarks(
                    display_frame,
                    landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )

                # 计算角度
                left_knee = self._calc_knee_angle(landmarks.landmark, 'left')
                right_knee = self._calc_knee_angle(landmarks.landmark, 'right')

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

        # ======================
        # 弹出窗口显示（关键！）
        # ======================
        if self.show_window:
            cv2.imshow("Face + Pose Detection", display_frame)
            cv2.waitKey(1)  # 必须加，否则窗口不刷新

        return frame_result

    def get_final_stats(self, total_frames: int, fps: float) -> Dict:
        """获取最终结果"""
        # 略（保持你原来逻辑不变）
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
        try:
            if side == 'left':
                hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
                knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
                ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            else:
                hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
                knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
                ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]

            return self._angle_3d([hip.x, hip.y], [knee.x, knee.y], [ankle.x, ankle.y])
        except:
            return None

    def _angle_3d(self, a, b, c) -> float:
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    @staticmethod
    def _get_cadence_level(cadence: float) -> str:
        if cadence < 80: return "慢走"
        elif cadence < 110: return "正常行走"
        elif cadence < 140: return "慢跑"
        else: return "快速跑步"

    @staticmethod
    def _get_symmetry_status(symmetry: float) -> str:
        if symmetry >= 80: return "良好"
        elif symmetry >= 60: return "一般"
        else: return "较差"

    @staticmethod
    def _get_face_description(has_face: bool, face_coverage: float, max_faces: int) -> str:
        if not has_face: return "未检测到人脸"
        if max_faces == 1: return f"单人视频，人脸覆盖 {face_coverage}% 的帧"
        else: return f"多人视频（最多{max_faces}人），人脸覆盖 {face_coverage}% 的帧"