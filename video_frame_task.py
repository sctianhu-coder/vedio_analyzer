from dataclasses import dataclass
import numpy as np
from datetime import datetime

@dataclass
class VideoFrameTask:
    task_id: str  # 唯一标识（如：视频ID_帧ID_分析类型）
    video_id: str  # 关联原始视频
    frame_id: int  # 帧序号
    frame_data: np.ndarray  # 帧像素数据（压缩/原始可选）
    timestamp: datetime  # 帧时间戳
    analysis_type: list  # 需执行的分析类型（["face", "gait", "fluency", "sensitive"]）
    priority: int = 0  # 任务优先级（可选）