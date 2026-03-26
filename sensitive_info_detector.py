"""
敏感信息识别模块
检测视频中的敏感内容，包括：
1. 人脸隐私（是否有人脸、是否需模糊处理）
2. 涉黄内容检测
3. 暴力内容检测
4. 敏感词检测（字幕、语音转录文本）
5. 个人身份信息（身份证、电话号码、银行卡号等）
"""

import re
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import hashlib


class SensitiveInfoDetector:
    """
    敏感信息检测器

    功能：
    1. 人脸检测（用于隐私保护）
    2. 涉黄/暴力内容检测（基于肤色分析和动作检测）
    3. 文本敏感词检测
    4. 个人身份信息检测
    """

    def __init__(self):
        """初始化检测器"""
        # 敏感词库
        self.sensitive_words = self._load_sensitive_words()

        # 个人身份信息正则表达式
        self.pii_patterns = {
            'phone': re.compile(r'1[3-9]\d{9}'),  # 手机号
            'id_card': re.compile(r'[1-9]\d{5}(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]'),  # 身份证
            'email': re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),  # 邮箱
            'bank_card': re.compile(r'\d{16,19}'),  # 银行卡号
            'ip_address': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),  # IP地址
        }

        # 涉黄/暴力关键词
        self.adult_keywords = [
            '色情', '成人', '情色', '裸露', '性爱',
            'porn', 'adult', 'nude', 'sex', 'xxx'
        ]

        self.violence_keywords = [
            '暴力', '血腥', '杀人', '殴打', '恐怖袭击',
            'violence', 'blood', 'kill', 'attack', 'terror'
        ]

    def _load_sensitive_words(self) -> List[str]:
        """
        加载敏感词库
        实际使用时可以从文件加载或接入外部敏感词API
        """
        return [
            # 政治敏感词（示例）
            '敏感词1', '敏感词2',
            # 涉黄敏感词
            '色情', '裸体', '性爱',
            # 暴力敏感词
            '暴力', '血腥', '恐怖',
            # 违法敏感词
            '毒品', '赌博', '诈骗',
        ]

    def detect_faces_for_privacy(self, frame: np.ndarray) -> Dict:
        """
        检测人脸用于隐私保护

        Args:
            frame: 图像帧

        Returns:
            人脸检测结果
        """
        try:
            import mediapipe as mp
            mp_face_detection = mp.solutions.face_detection

            with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)

                faces = []
                if results.detections:
                    h, w = frame.shape[:2]
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)

                        faces.append({
                            'bbox': (x, y, width, height),
                            'confidence': detection.score[0]
                        })

                return {
                    'has_faces': len(faces) > 0,
                    'face_count': len(faces),
                    'faces': faces,
                    'privacy_risk': len(faces) > 0,
                    'risk_level': 'high' if len(faces) > 5 else 'medium' if len(faces) > 0 else 'low'
                }
        except Exception as e:
            return {'has_faces': False, 'error': str(e)}

    def detect_skin_content(self, frame: np.ndarray) -> Dict:
        """
        检测裸露内容（基于肤色检测）

        Args:
            frame: 图像帧

        Returns:
            裸露检测结果
        """
        try:
            # 转换到 HSV 色彩空间
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # 肤色范围（HSV）
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)

            # 创建肤色掩码
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

            # 计算肤色比例
            skin_ratio = np.sum(skin_mask > 0) / (frame.shape[0] * frame.shape[1])

            # 判断是否可能包含裸露内容
            is_risky = skin_ratio > 0.3  # 肤色超过30%

            return {
                'skin_ratio': round(skin_ratio * 100, 2),
                'is_risky': is_risky,
                'risk_level': 'high' if skin_ratio > 0.5 else 'medium' if skin_ratio > 0.3 else 'low',
                'description': f'肤色占比 {skin_ratio * 100:.1f}%'
            }
        except Exception as e:
            return {'skin_ratio': 0, 'is_risky': False, 'error': str(e)}

    def detect_motion_abnormal(self, frame_diff: float) -> Dict:
        """
        检测异常运动（可能包含暴力行为）

        Args:
            frame_diff: 帧间差异值

        Returns:
            运动异常检测结果
        """
        # 帧间差异过大可能表示剧烈运动
        is_abnormal = frame_diff > 50

        return {
            'is_abnormal_motion': is_abnormal,
            'motion_intensity': round(frame_diff, 2),
            'risk_level': 'high' if frame_diff > 100 else 'medium' if frame_diff > 50 else 'low'
        }

    def detect_text_sensitive(self, text: str) -> Dict:
        """
        检测文本中的敏感信息

        Args:
            text: 文本内容

        Returns:
            敏感词检测结果
        """
        if not text:
            return {'has_sensitive': False, 'sensitive_words': [], 'risk_level': 'low'}

        found_words = []
        for word in self.sensitive_words:
            if word.lower() in text.lower():
                found_words.append(word)

        # 去重
        found_words = list(set(found_words))

        return {
            'has_sensitive': len(found_words) > 0,
            'sensitive_words': found_words,
            'word_count': len(found_words),
            'risk_level': 'high' if len(found_words) > 3 else 'medium' if len(found_words) > 0 else 'low'
        }

    def detect_pii(self, text: str) -> Dict:
        """
        检测个人身份信息（PII）

        Args:
            text: 文本内容

        Returns:
            PII检测结果
        """
        if not text:
            return {'has_pii': False, 'pii_types': [], 'risk_level': 'low'}

        found_pii = []

        for pii_type, pattern in self.pii_patterns.items():
            matches = pattern.findall(text)
            if matches:
                found_pii.append({
                    'type': pii_type,
                    'count': len(matches),
                    'examples': matches[:3]  # 只显示前3个示例
                })

        return {
            'has_pii': len(found_pii) > 0,
            'pii_types': [p['type'] for p in found_pii],
            'pii_details': found_pii,
            'risk_level': 'high' if len(found_pii) > 0 else 'low'
        }

    def analyze_frame(self, frame: np.ndarray, frame_diff: float = 0) -> Dict:
        """
        分析单帧的敏感信息

        Args:
            frame: 图像帧
            frame_diff: 帧间差异

        Returns:
            敏感信息分析结果
        """
        # 人脸隐私检测
        face_result = self.detect_faces_for_privacy(frame)

        # 肤色检测
        skin_result = self.detect_skin_content(frame)

        # 运动异常检测
        motion_result = self.detect_motion_abnormal(frame_diff)

        # 综合风险评估
        risk_scores = {
            'face': 3 if face_result.get('face_count', 0) > 5 else 2 if face_result.get('face_count', 0) > 0 else 0,
            'skin': 3 if skin_result.get('risk_level') == 'high' else 2 if skin_result.get(
                'risk_level') == 'medium' else 0,
            'motion': 3 if motion_result.get('risk_level') == 'high' else 2 if motion_result.get(
                'risk_level') == 'medium' else 0
        }

        total_risk = sum(risk_scores.values())

        if total_risk >= 6:
            overall_risk = 'high'
        elif total_risk >= 3:
            overall_risk = 'medium'
        else:
            overall_risk = 'low'

        return {
            'face_privacy': face_result,
            'skin_content': skin_result,
            'motion_abnormal': motion_result,
            'risk_scores': risk_scores,
            'overall_risk': overall_risk,
            'needs_blur': face_result.get('has_faces', False)  # 需要模糊人脸
        }

    def analyze_video_sensitive(
            self,
            video_path: str,
            transcript: str = None,
            frame_interval: int = 30
    ) -> Dict:
        """
        分析整个视频的敏感信息

        Args:
            video_path: 视频路径
            transcript: 语音转录文本（可选）
            frame_interval: 采样间隔（帧）

        Returns:
            敏感信息分析报告
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': '无法打开视频'}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 统计结果
        frame_results = []
        max_faces_per_frame = 0
        max_skin_ratio = 0
        abnormal_motion_frames = 0

        frame_count = 0
        prev_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 采样分析（每N帧分析一次，提高效率）
            if frame_count % frame_interval == 0:
                # 计算帧间差异
                frame_diff = 0
                if prev_frame is not None:
                    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame_diff = np.mean(np.abs(gray1 - gray2))

                # 分析当前帧
                result = self.analyze_frame(frame, frame_diff)
                frame_results.append(result)

                # 更新统计
                face_count = result['face_privacy'].get('face_count', 0)
                max_faces_per_frame = max(max_faces_per_frame, face_count)

                skin_ratio = result['skin_content'].get('skin_ratio', 0)
                max_skin_ratio = max(max_skin_ratio, skin_ratio)

                if result['motion_abnormal'].get('is_abnormal_motion', False):
                    abnormal_motion_frames += 1

                prev_frame = frame.copy()

        cap.release()

        # 文本敏感信息检测
        text_sensitive = None
        pii_detection = None

        if transcript:
            text_sensitive = self.detect_text_sensitive(transcript)
            pii_detection = self.detect_pii(transcript)

        # 计算高风险帧比例
        high_risk_frames = sum(1 for r in frame_results if r['overall_risk'] == 'high')
        risk_ratio = (high_risk_frames / len(frame_results) * 100) if frame_results else 0

        # 综合评估
        if risk_ratio > 30 or max_skin_ratio > 50:
            overall_risk = 'high'
        elif risk_ratio > 10 or max_skin_ratio > 30:
            overall_risk = 'medium'
        else:
            overall_risk = 'low'

        return {
            'status': 'success',
            'frame_analysis': {
                'sampled_frames': len(frame_results),
                'high_risk_frames': high_risk_frames,
                'risk_ratio': round(risk_ratio, 2),
                'max_faces_per_frame': max_faces_per_frame,
                'max_skin_ratio': round(max_skin_ratio, 2),
                'abnormal_motion_frames': abnormal_motion_frames
            },
            'text_analysis': {
                'has_transcript': transcript is not None,
                'text_sensitive': text_sensitive,
                'pii_detection': pii_detection
            },
            'overall_risk': overall_risk,
            'recommendations': self._generate_recommendations(overall_risk, max_faces_per_frame, max_skin_ratio),
            'needs_privacy_blur': max_faces_per_frame > 0,
            'needs_content_filter': overall_risk in ['medium', 'high']
        }

    def _generate_recommendations(self, overall_risk: str, max_faces: int, max_skin: float) -> List[str]:
        """生成处理建议"""
        recommendations = []

        if max_faces > 0:
            recommendations.append(f"检测到{max_faces}张人脸，建议进行人脸模糊处理以保护隐私")

        if max_skin > 30:
            recommendations.append("检测到较高比例的肤色区域，可能存在裸露内容，建议审核")

        if overall_risk == 'high':
            recommendations.append("视频存在高风险内容，建议立即下架或进行内容过滤")
        elif overall_risk == 'medium':
            recommendations.append("视频存在中等风险内容，建议人工审核")

        if not recommendations:
            recommendations.append("视频内容安全，无明显敏感信息")

        return recommendations


class PrivacyBlur:
    """隐私保护工具：人脸模糊处理"""

    @staticmethod
    def blur_faces(frame: np.ndarray, faces: List[Dict], kernel_size: Tuple[int, int] = (25, 25)) -> np.ndarray:
        """
        对检测到的人脸进行模糊处理

        Args:
            frame: 原始图像
            faces: 人脸列表
            kernel_size: 模糊核大小

        Returns:
            模糊处理后的图像
        """
        result = frame.copy()

        for face in faces:
            x, y, w, h = face['bbox']
            # 确保边界在图像内
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)

            if w > 0 and h > 0:
                # 提取人脸区域
                face_roi = result[y:y + h, x:x + w]
                # 高斯模糊
                blurred = cv2.GaussianBlur(face_roi, kernel_size, 0)
                result[y:y + h, x:x + w] = blurred

        return result