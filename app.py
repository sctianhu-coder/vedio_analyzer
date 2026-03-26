#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
视频分析系统 - 主程序
整合视频流畅度分析、步态识别、人脸检测、声音识别、敏感信息检测
"""

import os
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Optional, Any
import sys

# 导入 cv2
import cv2
import numpy as np

# 导入工具函数
from tools import convert_to_serializable, ensure_dir, format_duration

# 导入独立的分析模块
from video_smoothness import VideoSmoothnessAnalyzer
from gait_face_analyzer import GaitFaceAnalyzer
from audio_analyzer import AudioAnalyzer, AudioAnalyzerSimple
from sensitive_info_detector import SensitiveInfoDetector, PrivacyBlur


class VideoAnalysisApp:
    """
    视频分析应用主类
    整合流畅度分析、步态识别、人脸检测、声音识别、敏感信息检测
    """

    def __init__(self, enable_audio: bool = True, use_advanced_asr: bool = False,
                 enable_sensitive: bool = True, ffmpeg_path: str = None):
        """
        初始化应用

        Args:
            enable_audio: 是否启用音频分析
            use_advanced_asr: 是否使用高级语音识别
            enable_sensitive: 是否启用敏感信息检测
            ffmpeg_path: ffmpeg 可执行文件路径
        """
        self.smoothness_analyzer = VideoSmoothnessAnalyzer()
        self.gait_face_analyzer = GaitFaceAnalyzer()

        self.audio_analyzer = None
        if enable_audio:
            if use_advanced_asr:
                self.audio_analyzer = AudioAnalyzer(use_whisper=True)
            else:
                self.audio_analyzer = AudioAnalyzerSimple(ffmpeg_path=ffmpeg_path)

        self.sensitive_detector = SensitiveInfoDetector() if enable_sensitive else None
        self.privacy_blur = PrivacyBlur() if enable_sensitive else None

    def analyze_video(
            self,
            video_path: str,
            progress_callback: Optional[callable] = None,
            verbose: bool = True,
            analyze_audio: bool = True,
            detect_sensitive: bool = True
    ) -> Dict:
        """
        分析视频

        Args:
            video_path: 视频文件路径
            progress_callback: 进度回调函数
            verbose: 是否打印详细信息
            analyze_audio: 是否分析音频
            detect_sensitive: 是否检测敏感信息

        Returns:
            完整的分析结果字典
        """
        # 重置分析器
        self.smoothness_analyzer.reset()
        self.gait_face_analyzer.reset()

        # 检查文件是否存在
        if not os.path.exists(video_path):
            return {"status": "error", "message": f"视频文件不存在: {video_path}"}

        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"status": "error", "message": f"无法打开视频文件: {video_path}"}

        # 获取视频基本信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        if verbose:
            print(f"视频信息: {width}x{height}, {fps:.1f} fps, {total_frames}帧, {duration:.1f}秒")

        # 进度阶段
        stages = [
            (0.0, "读取视频帧"),
            (0.2, "检测人体姿态和人脸"),
            (0.4, "分析步态特征"),
            (0.6, "计算视频流畅度"),
            (0.8, "生成报告"),
            (1.0, "完成")
        ]

        frame_count = 0
        frame_diffs = []

        # 逐帧处理
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = time.time()

            # 更新进度
            if frame_count % 30 == 0 and progress_callback:
                progress = frame_count / total_frames
                for stage_progress, stage_text in stages:
                    if progress <= stage_progress:
                        progress_callback(progress, stage_text)
                        break

            # 1. 流畅度分析
            smooth_result = self.smoothness_analyzer.evaluate_frame(frame, current_time)
            frame_diffs.append(smooth_result.get('frame_diff', 0))

            # 2. 步态和人脸分析
            self.gait_face_analyzer.process_frame(frame)

        # 释放视频资源
        cap.release()

        # 获取视频分析统计结果
        smoothness_stats = self.smoothness_analyzer.get_final_stats()
        gait_face_stats = self.gait_face_analyzer.get_final_stats(total_frames, fps)

        # 3. 音频分析
        audio_result = {}
        transcript = None
        if analyze_audio and self.audio_analyzer:
            if verbose:
                print("\n正在分析音频...")

            audio_result = self.audio_analyzer.analyze_video_audio(video_path)
            transcript = audio_result.get('transcript', '')

            if verbose and audio_result.get("has_audio_content"):
                print(f"  音频: 有声音")
                if audio_result.get("has_voice"):
                    print(f"  人声: 检测到")
                    if transcript:
                        print(f"  转录: {transcript[:100]}...")

        # 4. 敏感信息检测
        sensitive_result = {}
        if detect_sensitive and self.sensitive_detector:
            if verbose:
                print("\n正在检测敏感信息...")

            # 分析视频敏感信息
            sensitive_result = self.sensitive_detector.analyze_video_sensitive(
                video_path,
                transcript=transcript,
                frame_interval=30
            )

            if verbose:
                risk = sensitive_result.get('overall_risk', 'unknown')
                print(f"  整体风险等级: {risk}")
                if sensitive_result.get('frame_analysis', {}).get('max_faces_per_frame', 0) > 0:
                    print(f"  检测到人脸: {sensitive_result['frame_analysis']['max_faces_per_frame']}张")
                if sensitive_result.get('frame_analysis', {}).get('max_skin_ratio', 0) > 30:
                    print(f"  肤色比例: {sensitive_result['frame_analysis']['max_skin_ratio']:.1f}%")

        # 构建完整结果
        result = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "video_info": {
                "file_path": os.path.abspath(video_path),
                "file_name": os.path.basename(video_path),
                "width": width,
                "height": height,
                "fps": round(fps, 2),
                "total_frames": total_frames,
                "duration_seconds": round(duration, 2),
                "duration_formatted": format_duration(duration)
            },
            "smoothness_analysis": smoothness_stats,
            "gait_analysis": gait_face_stats["gait_analysis"],
            "face_detection": gait_face_stats["face_detection"],
            "audio_analysis": audio_result,
            "sensitive_detection": sensitive_result,
            "summary": self._generate_summary(smoothness_stats, gait_face_stats["gait_analysis"],
                                              gait_face_stats["face_detection"], audio_result, sensitive_result)
        }

        return result

    def _generate_summary(self, smoothness_stats: Dict, gait_stats: Dict,
                          face_stats: Dict, audio_result: Dict, sensitive_result: Dict) -> str:
        """生成分析总结"""
        summary = []

        # 流畅度总结
        score = smoothness_stats['overall_score']
        if score >= 75:
            summary.append(f"视频流畅度良好({score:.0f}分)")
        else:
            summary.append(f"视频流畅度待提升({score:.0f}分)")

        # 步态总结
        if gait_stats['step_count'] > 0:
            summary.append(f"检测到{gait_stats['step_count']}步，步频{gait_stats['cadence']:.0f}步/分钟")

            if gait_stats['knee_angles']['symmetry'] >= 80:
                summary.append("左右腿步态对称")
            else:
                summary.append("左右腿步态存在不对称")

        # 人脸总结
        if face_stats['has_face']:
            summary.append(f"视频中包含人脸，覆盖{face_stats['face_coverage']}%的帧")
        else:
            summary.append("未检测到人脸")

        # 声音总结
        if audio_result.get("has_audio_content"):
            if audio_result.get("has_voice"):
                summary.append("检测到人声")
                if audio_result.get("transcript"):
                    transcript = audio_result['transcript'][:50]
                    summary.append(f"语音内容: {transcript}...")
            else:
                summary.append("有背景音但无人声")
        else:
            summary.append("视频无声")

        # 敏感信息总结
        if sensitive_result:
            risk = sensitive_result.get('overall_risk', 'low')
            if risk == 'high':
                summary.append("高风险内容，建议审核")
            elif risk == 'medium':
                summary.append("中等风险内容，建议人工审核")
            else:
                summary.append("内容安全")

        return "；".join(summary)


def print_progress(progress: float, status: str):
    """打印进度信息"""
    bar_length = 40
    filled = int(bar_length * progress)
    bar = '=' * filled + '-' * (bar_length - filled)
    print(f"\r[{bar}] {int(progress * 100)}% - {status}", end='', flush=True)


def save_result_to_json(result: Dict, output_dir: str = "output", output_file: str = None) -> str:
    """保存分析结果到JSON文件（自动转换NumPy类型）"""
    # 确保输出目录存在
    ensure_dir(output_dir)

    if output_file:
        json_file = os.path.join(output_dir, output_file)
    else:
        video_name = result['video_info']['file_name']
        base_name = os.path.splitext(video_name)[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_file = os.path.join(output_dir, f"{base_name}_{timestamp}.json")

    # 转换 NumPy 类型为 Python 原生类型
    serializable_result = convert_to_serializable(result)

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, ensure_ascii=False, indent=2)

    return json_file


def print_json_result(result: Dict):
    """打印JSON格式的结果（自动转换NumPy类型）"""
    print("\n" + "=" * 60)
    print("分析结果 (JSON格式)")
    print("=" * 60)

    # 转换 NumPy 类型为 Python 原生类型
    serializable_result = convert_to_serializable(result)

    print(json.dumps(serializable_result, ensure_ascii=False, indent=2))


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='视频分析系统 - 分析视频的流畅度、步态、人脸和声音',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基础分析
  python app.py -i video/running.mp4

  # 禁用音频分析
  python app.py -i video/running.mp4 --no-audio

  # 使用高级语音识别（Whisper）
  python app.py -i video/running.mp4 --asr

  # 指定 ffmpeg 路径
  python app.py -i video/running.mp4 --ffmpeg-path /opt/homebrew/bin/ffmpeg

  # 指定输出文件
  python app.py -i video/running.mp4 -o result.json

  # 静默模式
  python app.py -i video/running.mp4 -q
        """
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='输入视频文件路径'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='输出JSON文件路径'
    )

    parser.add_argument(
        '-d', '--output-dir',
        type=str,
        default='output',
        help='输出目录（默认: output）'
    )

    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='静默模式'
    )

    parser.add_argument(
        '--no-audio',
        action='store_true',
        help='禁用音频分析'
    )

    parser.add_argument(
        '--asr',
        action='store_true',
        help='使用高级语音识别（Whisper）'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='只打印结果，不保存文件'
    )

    parser.add_argument(
        '--ffmpeg-path',
        type=str,
        default=None,
        help='ffmpeg 可执行文件路径（例如: /opt/homebrew/bin/ffmpeg）'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()

    video_path = args.input

    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在 - {video_path}")
        sys.exit(1)

    if not args.quiet:
        print("=" * 60)
        print("视频分析系统 - 流畅度 + 步态 + 人脸 + 声音")
        print("=" * 60)
        print(f"输入文件: {video_path}")
        print(f"音频分析: {'禁用' if args.no_audio else '启用'}")
        print(f"语音识别: {'高级(Whisper)' if args.asr else '基础'}")
        if args.ffmpeg_path:
            print(f"ffmpeg路径: {args.ffmpeg_path}")
        print()

    # 初始化分析器，传入 ffmpeg_path
    app = VideoAnalysisApp(
        enable_audio=not args.no_audio,
        use_advanced_asr=args.asr,
        ffmpeg_path=args.ffmpeg_path
    )

    progress_callback = None if args.quiet else print_progress

    if not args.quiet:
        print("正在分析...")

    result = app.analyze_video(
        video_path,
        progress_callback,
        verbose=not args.quiet,
        analyze_audio=not args.no_audio
    )

    if not args.quiet and progress_callback:
        print()

    if result.get("status") == "error":
        print(f"\n错误: {result.get('message')}")
        sys.exit(1)

    print_json_result(result)

    if not args.no_save:
        json_file = save_result_to_json(
            result,
            output_dir=args.output_dir,
            output_file=args.output
        )
        print(f"\n结果已保存到: {json_file}")


if __name__ == "__main__":
    main()