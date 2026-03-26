**使用示例:
  # 基础分析
  python app.py -i video/running.mp4 --ffmpeg-path /opt/homebrew/bin/ffmpeg

  # 禁用音频分析
  python app.py -i video/running.mp4 --no-audio

  # 使用高级语音识别（Whisper）
  python app.py -i video/running.mp4 --asr

  # 指定输出文件
  python app.py -i video/running.mp4 -o result.json

  # 静默模式
  python app.py -i video/running.mp4 -q

**代码结构说明：

  ./video_analyzer
  +----app.py 主入库
  +----audio_analyzer.py 做声音分析的类
  +----gait_face_analyzer.py 人脸和步态提取的类
  +----sensitive_info_detector.py 敏感词汇提取的类，敏感词里面做了些简单枚举
  +----video_smoothness.py 视频流畅和平滑度检查的类
  +----tools.py                    # 工具函数（新增）

## 系统要求
- Python 3.12 -代码3.12上运行OK
- ffmpeg 需要本机有安装，不再环境变量中需要通过启动参数 传入，如：--ffmpeg-path /opt/homebrew/bin/ffmpeg

**结果输出说明，
    最后输出结果会在./output/result.json中
    ============================================================
    分析结果 (JSON格式)
    ============================================================
    {
      "status": "success",
      "timestamp": "2026-03-26T19:42:59.420329",
      "video_info": {
        "file_path": "/Users/hutian/Downloads/running.mp4",
        "file_name": "running.mp4",
        "width": 1280,
        "height": 720,
        "fps": 25.0,
        "total_frames": 3166,
        "duration_seconds": 126.64,
        "duration_formatted": "2:06"
      },
      "smoothness_analysis": {
        "overall_score": 93.0,
        "score_level": "极佳",
        "motion_smoothness": 0.0,
        "avg_fps": 5.3,
        "fps_std": 0.07,
        "stutter_frames": 7,
        "freeze_frames": 0,
        "is_smooth": true
      },
      "gait_analysis": {
        "step_count": 1015,
        "cadence": 539.7,
        "cadence_level": "快速跑步",
        "knee_angles": {
          "left": 145.8,
          "right": 135.2,
          "symmetry": 88.2,
          "symmetry_status": "良好"
        },
        "detection_rate": 89.1
      },
      "face_detection": {
        "has_face": true,
        "face_coverage": 0.4,
        "face_detected_frames": 14,
        "total_faces_detected": 14,
        "max_faces_in_frame": 1,
        "avg_faces_per_frame": 0.0,
        "description": "单人视频，人脸覆盖 0.4421983575489577% 的帧"
      },
      "audio_analysis": {
        "has_audio_stream": false,
        "has_audio_content": false,
        "has_voice": false,
        "transcript": "",
        "error": "无法提取音频（ffmpeg 未安装或提取失败）"
      },
      "sensitive_detection": {
        "status": "success",
        "frame_analysis": {
          "sampled_frames": 105,
          "high_risk_frames": 0,
          "risk_ratio": 0.0,
          "max_faces_per_frame": 1,
          "max_skin_ratio": 28.0,
          "abnormal_motion_frames": 103
        },
        "text_analysis": {
          "has_transcript": true,
          "text_sensitive": null,
          "pii_detection": null
        },
        "overall_risk": "low",
        "recommendations": [
          "检测到1张人脸，建议进行人脸模糊处理以保护隐私"
        ],
        "needs_privacy_blur": true,
        "needs_content_filter": false
      },
      "summary": "视频流畅度良好(93分)；检测到1015步，步频540步/分钟；左右腿步态对称；视频中包含人脸，覆盖0.4%的帧；视频无声；内容安全"
    }

    结果已保存到: output/./result.json

**日志打印说明：
    1. 运行中会打印处理进入如下例子
        [=========-------------------------------] 23% - 分析步态特征
    2.过程中第三方组件会打印，如下日志，请忽略
        0000 00:00:1774524916.434385 20188219 gl_context.cc:357] GL version: 2.1 (2.1 Metal - 89.4), renderer: Apple M1
    3.启动时候会打印视频基本信息，如下格式
        ============================================================
        视频分析系统 - 流畅度 + 步态 + 人脸 + 声音
        ============================================================
        输入文件: /Users/hutian/Downloads/running.mp4
        音频分析: 启用
        语音识别: 基础

        正在分析...
        视频信息: 1280x720, 25.0 fps, 3166帧, 126.6秒


FROM registry.cn-beijing.aliyuncs.com/aliyunfc/runtime-python3.10:latest