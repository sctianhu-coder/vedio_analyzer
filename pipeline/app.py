#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视频分析系统 - CLI入口
使用 Pipeline 架构
"""

import os
import sys
import argparse
import json
from datetime import datetime

from analyzer import VideoAnalysisCore
import tools


def print_progress(progress: float, status: str = "处理中"):
    """CLI进度条打印"""
    bar_length = 40
    filled = int(bar_length * progress / 100)
    bar = '=' * filled + '-' * (bar_length - filled)
    print(f"\r[{bar}] {progress:.1f}% - {status}", end='', flush=True)


def parse_arguments():
    """参数解析"""
    parser = argparse.ArgumentParser(
        description='视频分析系统 - 分析视频的流畅度、步态、人脸、声音和敏感信息',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基础分析（完整分析）
  python app.py -i video.mp4

  # 禁用音频分析
  python app.py -i video.mp4 --no-audio

  # 禁用敏感信息检测
  python app.py -i video.mp4 --no-sensitive

  # 静默模式（不打印详细日志）
  python app.py -i video.mp4 -q

  # 指定输出目录
  python app.py -i video.mp4 -d ./my_results

  # 指定输出文件
  python app.py -i video.mp4 -o result.json
        """
    )

    # 输入参数
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='输入视频文件路径'
    )

    # 输出参数
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

    # 分析选项
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
        '--no-sensitive',
        action='store_true',
        help='禁用敏感信息检测'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='只打印结果，不保存文件'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=2,
        help='多线程处理的最大线程数（默认: 2）'
    )

    # 兼容旧参数（忽略）
    parser.add_argument('--asr', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--submit', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--status', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--result', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--batch', type=str, nargs='+', help=argparse.SUPPRESS)
    parser.add_argument('--daemon', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--max-concurrent', type=int, default=1, help=argparse.SUPPRESS)

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    video_path = args.input

    # 检查文件是否存在
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在 - {video_path}")
        sys.exit(1)

    # 打印基本信息
    if not args.quiet:
        print("=" * 60)
        print("视频分析系统 - 流畅度 + 步态 + 人脸 + 声音 + 敏感信息检测")
        print("=" * 60)
        print(f"输入文件: {video_path}")
        print(f"音频分析: {'禁用' if args.no_audio else '启用'}")
        print(f"敏感检测: {'禁用' if args.no_sensitive else '启用'}")
        print(f"输出目录: {args.output_dir}")
        print()

    # 初始化分析器
    core = VideoAnalysisCore(
        enable_audio=not args.no_audio,
        enable_sensitive=not args.no_sensitive,
        max_workers=args.max_workers
    )

    # 进度回调
    progress_callback = None
    if not args.quiet:
        def progress_cb(progress: float):
            print_progress(progress)

        progress_callback = progress_cb

    # 开始分析
    if not args.quiet:
        print("正在分析...")

    try:
        result = core.analyze_video(
            video_path=video_path,
            progress_callback=progress_callback,
            verbose=not args.quiet,
            analyze_audio=not args.no_audio,
            detect_sensitive=not args.no_sensitive
        )
    except Exception as e:
        print(f"\n分析过程异常: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 换行（进度条后）
    if not args.quiet and progress_callback:
        print()

    # 检查错误
    if result.get("status") == "error":
        print(f"\n错误: {result.get('message')}")
        sys.exit(1)

    # 打印结果
    if not args.quiet:
        print("\n分析结果:")
        serializable = tools.convert_to_serializable(result)
        print(json.dumps(serializable, ensure_ascii=False, indent=2))

    # 保存结果
    if not args.no_save:
        if args.output:
            # 指定输出文件路径
            output_dir = os.path.dirname(args.output) or args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            try:
                serializable = tools.convert_to_serializable(result)
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(serializable, f, ensure_ascii=False, indent=2)
                print(f"\n结果已保存到: {args.output}")
            except Exception as e:
                print(f"\n保存失败: {e}")
        else:
            # 使用默认保存方式
            json_file = core.save_result_to_json(result, output_dir=args.output_dir)
            if json_file:
                print(f"\n结果已保存到: {json_file}")
            else:
                print("\n⚠️ 结果保存失败")

    sys.exit(0)


if __name__ == "__main__":
    main()