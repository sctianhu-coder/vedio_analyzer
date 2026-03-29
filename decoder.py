import ffmpeg
import numpy as np
import subprocess
import json

class VideoDecoder:
    @staticmethod
    def decode(video_path: str):
        """
        同时解码视频帧 + 音频（全部内存，不写文件）
        返回：frames (np.array), audio_bytes (bytes)
        """
        # ============= 1. 获取视频信息 =============
        probe = ffmpeg.probe(video_path)
        video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        fps = eval(video_stream['r_frame_rate'])

        # ============= 2. 提取视频帧（内存管道）=============
        video_cmd = (
            ffmpeg
            .input(video_path)
            .output('pipe:', format='rawvideo', pix_fmt='bgr24')
            .compile()
        )
        video_out = subprocess.check_output(video_cmd)

        # ============= 3. 提取音频（内存管道）=============
        audio_cmd = (
            ffmpeg
            .input(video_path)
            .output('pipe:', format='wav')
            .compile()
        )
        audio_out = subprocess.check_output(audio_cmd)

        # ============= 4. 转换为帧格式 =============
        frame_size = width * height * 3
        frames = np.frombuffer(video_out, np.uint8).reshape((-1, height, width, 3))

        return frames, audio_out


# ===================== 测试 =====================
if __name__ == "__main__":
    VIDEO_PATH = "/Users/hutian/Downloads/running.avi"
    frames, audio_data = VideoDecoder.decode(VIDEO_PATH)

    print("✅ 解码成功！")
    print(f"总帧数: {len(frames)}")
    print(f"视频尺寸: {frames.shape}")
    print(f"音频大小: {len(audio_data)} bytes")