"""
音频分析模块
支持声音检测、语音识别、说话人识别等功能
"""

import os
import tempfile
import numpy as np
import subprocess
from typing import Dict, Optional, Any


class AudioAnalyzer:
    """
    完整音频分析器（使用 librosa、soundfile、moviepy）

    功能：
    1. 检测视频中是否有声音
    2. 检测是否有人声（语音）
    3. 语音转文字（ASR）
    """

    def __init__(self, use_whisper: bool = True):
        """
        初始化音频分析器

        Args:
            use_whisper: 是否使用 Whisper 进行语音识别
        """
        self.use_whisper = use_whisper
        self._init_audio_processing()

        # 尝试加载语音识别模型
        self.asr_model = None
        if use_whisper:
            self._load_whisper()

    def _init_audio_processing(self):
        """初始化音频处理库"""
        self.librosa = None
        self.sf = None

        try:
            import librosa
            self.librosa = librosa
        except ImportError:
            pass

        try:
            import soundfile as sf
            self.sf = sf
        except ImportError:
            pass

    def _load_whisper(self):
        """加载 Whisper 语音识别模型"""
        try:
            import whisper
            self.asr_model = whisper.load_model("base")
            print("Whisper 语音识别模型已加载")
        except ImportError:
            print("警告: openai-whisper 未安装，语音转文字功能不可用")
            self.use_whisper = False
        except Exception as e:
            print(f"Whisper 加载失败: {e}")
            self.use_whisper = False

    def extract_audio_from_video(self, video_path: str, output_audio_path: str = None) -> Optional[str]:
        """
        从视频中提取音频（使用 moviepy）
        """
        try:
            import moviepy.editor as mp
            video = mp.VideoFileClip(video_path)

            if output_audio_path is None:
                output_audio_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name

            if video.audio is not None:
                video.audio.write_audiofile(output_audio_path, verbose=False, logger=None)
                video.close()
                return output_audio_path
            else:
                print("视频没有音频轨道")
                return None
        except ImportError:
            print("警告: moviepy 未安装，无法提取音频")
            print("安装命令: pip install moviepy")
            return None
        except Exception as e:
            print(f"音频提取失败: {e}")
            return None

    def detect_audio_presence(self, audio_path: str) -> Dict:
        """检测音频文件中是否有声音"""
        if self.sf is None:
            return {"has_audio": False, "error": "soundfile not installed"}

        try:
            data, samplerate = self.sf.read(audio_path)
            energy = np.mean(data ** 2)
            is_silent = energy < 1e-8

            return {
                "has_audio": energy > 1e-6,
                "is_silent": is_silent,
                "energy": float(energy),
                "duration": len(data) / samplerate,
                "sample_rate": samplerate
            }
        except Exception as e:
            return {"has_audio": False, "error": str(e)}

    def detect_voice(self, audio_path: str) -> Dict:
        """检测音频中是否有人声"""
        if self.librosa is None:
            return {"has_voice": False, "error": "librosa not installed"}

        try:
            y, sr = self.librosa.load(audio_path, sr=16000)
            energy = np.mean(y ** 2)
            zcr = np.mean(self.librosa.feature.zero_crossing_rate(y))
            has_voice = energy > 1e-5 and zcr > 0.01

            voice_likelihood = self._estimate_voice_likelihood(y, sr)

            return {
                "has_voice": has_voice or voice_likelihood > 0.5,
                "voice_likelihood": float(voice_likelihood),
                "energy": float(energy),
                "duration": len(y) / sr
            }
        except Exception as e:
            return {"has_voice": False, "error": str(e)}

    def _estimate_voice_likelihood(self, y: np.ndarray, sr: int) -> float:
        """估计人声可能性"""
        try:
            mfcc = self.librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_std = np.std(mfcc, axis=1)
            voice_score = np.mean(mfcc_std) / 10.0

            harmonic, percussive = self.librosa.effects.hpss(y)
            harmonic_energy = np.mean(harmonic ** 2)
            percussive_energy = np.mean(percussive ** 2)
            harmonic_ratio = harmonic_energy / (percussive_energy + 1e-6)
            harmonic_score = min(1.0, harmonic_ratio / 2)

            likelihood = (voice_score * 0.6 + harmonic_score * 0.4)
            return min(1.0, max(0.0, likelihood))
        except Exception:
            return 0.0

    def transcribe_audio(self, audio_path: str) -> Dict:
        """语音转文字"""
        if not self.use_whisper or self.asr_model is None:
            return {"transcript": "", "error": "Whisper not available"}

        try:
            result = self.asr_model.transcribe(audio_path)
            return {
                "transcript": result["text"],
                "language": result.get("language", "unknown"),
                "segments": result.get("segments", [])
            }
        except Exception as e:
            return {"transcript": "", "error": str(e)}

    def analyze_video_audio(self, video_path: str) -> Dict:
        """综合分析视频音频"""
        result = {
            "has_audio_stream": False,
            "has_audio_content": False,
            "has_voice": False,
            "transcript": "",
            "error": None
        }

        audio_path = self.extract_audio_from_video(video_path)
        if audio_path is None:
            result["error"] = "无法提取音频（视频可能无音频轨道或moviepy未安装）"
            return result

        try:
            audio_presence = self.detect_audio_presence(audio_path)
            result["has_audio_stream"] = audio_presence.get("has_audio", False)
            result["has_audio_content"] = not audio_presence.get("is_silent", True)

            if result["has_audio_content"]:
                voice_result = self.detect_voice(audio_path)
                result["has_voice"] = voice_result.get("has_voice", False)
                result["voice_likelihood"] = voice_result.get("voice_likelihood", 0)

                if result["has_voice"] and self.use_whisper:
                    transcript_result = self.transcribe_audio(audio_path)
                    result["transcript"] = transcript_result.get("transcript", "")

            result["audio_info"] = audio_presence

        except Exception as e:
            result["error"] = str(e)

        finally:
            try:
                os.unlink(audio_path)
            except:
                pass

        return result


class AudioAnalyzerSimple:
    """
    轻量级音频分析器（使用 ffmpeg 或 moviepy 提取音频）
    支持外部传入 ffmpeg 路径
    """

    def __init__(self, ffmpeg_path: str = None):
        """
        初始化音频分析器

        Args:
            ffmpeg_path: ffmpeg 可执行文件的完整路径，如果不提供则自动查找
        """
        self.ffmpeg_path = ffmpeg_path
        self.ffmpeg_available = self._check_ffmpeg()
        self.moviepy_available = self._check_moviepy()

        if self.ffmpeg_available:
            print(f"ffmpeg 可用: {self.ffmpeg_path}")
        else:
            print("ffmpeg 不可用")

    def _find_ffmpeg(self) -> Optional[str]:
        """自动查找 ffmpeg 路径"""
        import shutil

        # 常见 ffmpeg 安装路径
        possible_paths = [
            'ffmpeg',  # 在 PATH 中
            '/opt/homebrew/bin/ffmpeg',  # Apple Silicon Mac
            '/usr/local/bin/ffmpeg',      # Intel Mac
            '/usr/bin/ffmpeg',             # Linux
            'C:\\ffmpeg\\bin\\ffmpeg.exe', # Windows
        ]

        for path in possible_paths:
            # 检查是否是完整路径
            if os.path.exists(path):
                return path
            # 检查是否在 PATH 中
            found = shutil.which(path)
            if found:
                return found

        return None

    def _check_ffmpeg(self) -> bool:
        """检查 ffmpeg 是否可用"""
        # 确定要使用的 ffmpeg 路径
        ffmpeg_cmd = self.ffmpeg_path if self.ffmpeg_path else self._find_ffmpeg()

        if not ffmpeg_cmd:
            return False

        try:
            result = subprocess.run(
                [ffmpeg_cmd, '-version'],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                # 更新 ffmpeg_path 为找到的路径
                self.ffmpeg_path = ffmpeg_cmd
                return True
        except Exception:
            pass

        return False

    def _check_moviepy(self) -> bool:
        """检查 moviepy 是否可用"""
        try:
            import moviepy.editor as mp
            return True
        except ImportError:
            return False

    def extract_audio_ffmpeg(self, video_path: str, output_path: str = None) -> Optional[str]:
        """使用 ffmpeg 直接提取音频"""
        if not self.ffmpeg_available or not self.ffmpeg_path:
            return None

        if output_path is None:
            output_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name

        cmd = [
            self.ffmpeg_path,  # 使用指定的 ffmpeg 路径
            '-i', video_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',
            output_path
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30
            )
            if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path
        except Exception:
            pass
        return None

    def extract_audio_moviepy(self, video_path: str, output_path: str = None) -> Optional[str]:
        """使用 moviepy 提取音频"""
        if not self.moviepy_available:
            return None

        if output_path is None:
            output_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name

        try:
            import moviepy.editor as mp
            video = mp.VideoFileClip(video_path)

            if video.audio is None:
                return None

            video.audio.write_audiofile(output_path, verbose=False, logger=None, fps=16000)
            video.close()

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path
        except Exception:
            pass
        return None

    def extract_audio(self, video_path: str, output_path: str = None) -> Optional[str]:
        """尝试多种方法提取音频"""
        # 优先使用 ffmpeg（更可靠）
        audio_path = self.extract_audio_ffmpeg(video_path, output_path)
        if audio_path:
            return audio_path

        # 备选使用 moviepy
        audio_path = self.extract_audio_moviepy(video_path, output_path)
        if audio_path:
            return audio_path

        return None

    def analyze_video_audio(self, video_path: str) -> dict:
        """综合分析视频音频"""
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

        # 提取音频
        audio_path = self.extract_audio(video_path)
        if audio_path is None:
            result["error"] = "无法提取音频"
            if not self.ffmpeg_available and not self.moviepy_available:
                result["message"] = "请安装 ffmpeg (brew install ffmpeg) 或 moviepy (pip install moviepy)"
            elif not self.ffmpeg_available:
                result["message"] = f"ffmpeg 不可用，请指定正确路径或安装 ffmpeg"
            else:
                result["message"] = "视频可能没有音频轨道"
            return result

        try:
            import wave

            with wave.open(audio_path, 'rb') as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()

                data = wav.readframes(frames)
                if len(data) == 0:
                    result["message"] = "音频数据为空"
                    return result

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
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
            except:
                pass

        return result

    def analyze_audio(self, video_path: str) -> dict:
        """简化版音频分析（保持向后兼容）"""
        return self.analyze_video_audio(video_path)


# 便捷函数
def quick_audio_check(video_path: str, ffmpeg_path: str = None) -> dict:
    """
    快速检查视频是否有声音和人声

    Args:
        video_path: 视频路径
        ffmpeg_path: ffmpeg 可执行文件路径（可选）

    Returns:
        音频分析结果
    """
    analyzer = AudioAnalyzerSimple(ffmpeg_path=ffmpeg_path)
    return analyzer.analyze_video_audio(video_path)