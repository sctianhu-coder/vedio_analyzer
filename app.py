#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视频分析系统 - CLI入口
保留完整参数解析、日志输出、结果保存逻辑
"""
import os
import sys
import argparse
from core import VideoAnalysisCore


def print_progress(progress: float, status: str = "处理中"):
    """CLI进度条打印"""
    bar_length = 40
    filled = int(bar_length * progress / 100)
    bar = '=' * filled + '-' * (bar_length - filled)
    print(f"\r[{bar}] {progress:.1f}% - {status}", end='', flush=True)


def parse_arguments():
    """完整保留原参数解析逻辑"""
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
        '--no-sensitive',
        action='store_true',
        help='禁用敏感信息检测'
    )
    parser.add_argument(
        '--ffmpeg-path',
        type=str,
        default=None,
        help='ffmpeg 可执行文件路径（例如: /opt/homebrew/bin/ffmpeg）'
    )
    return parser.parse_args()


def main():
    """CLI主函数（保留所有原逻辑）"""
    args = parse_arguments()
    video_path = args.input

    # 基础校验
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在 - {video_path}")
        sys.exit(1)

    # 打印基础信息（非静默模式）
    if not args.quiet:
        print("=" * 60)
        print("视频分析系统 - 流畅度 + 步态 + 人脸 + 声音 + 敏感信息检测")
        print("=" * 60)
        print(f"输入文件: {video_path}")
        print(f"音频分析: {'禁用' if args.no_audio else '启用'}")
        print(f"语音识别: {'高级(Whisper)' if args.asr else '基础'}")
        print(f"敏感检测: {'禁用' if args.no_sensitive else '启用'}")
        if args.ffmpeg_path:
            print(f"ffmpeg路径: {args.ffmpeg_path}")
        print()

    # 初始化核心分析器（完整透传参数）
    core = VideoAnalysisCore(
        enable_audio=not args.no_audio,
        use_advanced_asr=args.asr,
        enable_sensitive=not args.no_sensitive,
        ffmpeg_path=args.ffmpeg_path
    )

    # 进度回调（非静默模式）
    progress_callback = None
    if not args.quiet:
        def progress_cb(progress: float):
            print_progress(progress)
        progress_callback = progress_cb

    # 开始分析
    if not args.quiet:
        print("正在分析...")
    result = core.analyze_video(
        video_path=video_path,
        progress_callback=progress_callback,
        verbose=not args.quiet,
        analyze_audio=not args.no_audio,
        detect_sensitive=not args.no_sensitive
    )

    # 处理结果
    if not args.quiet and progress_callback:
        print()  # 进度条换行

    if result.get("status") == "error":
        print(f"\n错误: {result.get('message')}")
        sys.exit(1)

    # 打印JSON结果
    core.print_json_result(result)

    # 保存结果（非禁用保存）
    if not args.no_save:
        json_file = core.save_result_to_json(
            result,
            output_dir=args.output_dir,
            output_file=args.output
        )
        print(f"\n结果已保存到: {json_file}")


if __name__ == "__main__":
    main()