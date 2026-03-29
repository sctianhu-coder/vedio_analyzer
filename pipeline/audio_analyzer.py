"""
音频分析模块
支持声音检测、语音识别、说话人识别等功能
"""

import os
import tempfile
import numpy as np
import subprocess
import wave
import io
from typing import Dict, Optional, Any


class AudioAnalyzerSimple:
    """
    轻量级音频分析器（支持从内存bytes分析）
    """

    def __init__(self, ffmpeg_path: str = None):
        self.ffmpeg_path = ffmpeg_path or "ffmpeg"
        self.ffmpeg_available = self._check_ffmpeg()

        if self.ffmpeg_available:
            print(f"✅ ffmpeg 自动检测成功: {self.ffmpeg_path}")
        else:
            print("⚠️ ffmpeg 不可用")

    def _check_ffmpeg(self) -> bool:
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    # ========== 新增：从内存bytes分析音频 ==========
    def analyze_audio_from_bytes(self, audio_bytes: bytes) -> Dict:
        """
        直接从内存中的音频数据进行分析（无需写文件）

        Args:
            audio_bytes: WAV格式的音频字节数据

        Returns:
            音频分析结果字典
        """
        result = {
            "has_audio_stream": False,
            "has_audio_content": False,
            "has_voice": False,
            "transcript": "",
            "error": None,
            "message": ""
        }

        if not audio_bytes or len(audio_bytes) < 44:  # WAV头至少44字节
            result["error"] = "音频数据为空或无效"
            result["message"] = "视频可能无音频轨道"
            return result

        try:
            # 使用 io.BytesIO 从内存读取
            with io.BytesIO(audio_bytes) as audio_buffer:
                with wave.open(audio_buffer, 'rb') as wav:
                    frames = wav.getnframes()
                    rate = wav.getframerate()
                    data = wav.readframes(frames)

                    if wav.getsampwidth() == 2:
                        audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    else:
                        audio_array = np.frombuffer(data, dtype=np.uint8).astype(np.float32) / 255.0

                    energy = np.mean(audio_array ** 2)

                    # 语音检测（基于频域）
                    fft = np.fft.rfft(audio_array)
                    freqs = np.fft.rfftfreq(len(audio_array), 1 / rate)
                    magnitude = np.abs(fft)

                    # 人声频率范围 85-255Hz
                    voice_band = (freqs >= 85) & (freqs <= 255)
                    voice_energy = np.sum(magnitude[voice_band])
                    total_energy = np.sum(magnitude)
                    voice_ratio = voice_energy / (total_energy + 1e-6)
                    has_voice = energy > 0.01 and voice_ratio > 0.3

                    result["has_audio_stream"] = True
                    result["has_audio_content"] = not (energy < 1e-6)
                    result["has_voice"] = has_voice
                    result["voice_confidence"] = float(min(1.0, voice_ratio))
                    result["audio_info"] = {
                        "duration": frames / rate,
                        "sample_rate": rate,
                        "energy": float(energy),
                        "voice_ratio": float(voice_ratio)
                    }
                    result["message"] = "音频分析完成"

        except Exception as e:
            result["error"] = str(e)

        return result

    # 保留原有方法用于兼容（从文件路径分析）
    def analyze_video_audio(self, video_path: str) -> Dict:
        """从文件路径分析音频（兼容旧接口）"""
        result = {
            "has_audio_stream": False,
            "has_audio_content": False,
            "has_voice": False,
            "transcript": "",
            "error": None,
            "message": ""
        }

        if not os.path.exists(video_path):
            result["error"] = f"视频文件不存在: {video_path}"
            return result

        audio_path = self.extract_audio(video_path)
        if audio_path is None:
            result["error"] = "无法提取音频"
            result["message"] = "请安装 ffmpeg"
            return result

        try:
            with wave.open(audio_path, 'rb') as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                data = wav.readframes(frames)

                if wav.getsampwidth() == 2:
                    audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    audio_array = np.frombuffer(data, dtype=np.uint8).astype(np.float32) / 255.0

                energy = np.mean(audio_array ** 2)

                fft = np.fft.rfft(audio_array)
                freqs = np.fft.rfftfreq(len(audio_array), 1 / rate)
                magnitude = np.abs(fft)

                voice_band = (freqs >= 85) & (freqs <= 255)
                voice_energy = np.sum(magnitude[voice_band])
                total_energy = np.sum(magnitude)
                voice_ratio = voice_energy / (total_energy + 1e-6)
                has_voice = energy > 0.01 and voice_ratio > 0.3

                result["has_audio_stream"] = True
                result["has_audio_content"] = not (energy < 1e-6)
                result["has_voice"] = has_voice
                result["voice_confidence"] = float(min(1.0, voice_ratio))
                result["audio_info"] = {
                    "duration": frames / rate,
                    "sample_rate": rate,
                    "energy": float(energy),
                    "voice_ratio": float(voice_ratio)
                }
                result["message"] = "音频分析完成"

        except Exception as e:
            result["error"] = str(e)

        finally:
            try:
                os.unlink(audio_path)
            except:
                pass

        return result

    def extract_audio(self, video_path: str, output_path: str = None) -> Optional[str]:
        """从视频文件提取音频到临时文件"""
        if not self.ffmpeg_available:
            return None

        if output_path is None:
            output_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name

        cmd = [
            self.ffmpeg_path,
            '-i', video_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',
            output_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            if result.returncode == 0 and os.path.getsize(output_path) > 0:
                return output_path
        except Exception:
            pass
        return None

    def analyze_audio(self, video_path: str) -> Dict:
        return self.analyze_video_audio(video_path)


def quick_audio_check(video_path: str, ffmpeg_path: str = None) -> dict:
    analyzer = AudioAnalyzerSimple(ffmpeg_path=ffmpeg_path)
    return analyzer.analyze_video_audio(video_path)