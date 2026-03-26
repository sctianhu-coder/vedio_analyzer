#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
工具函数模块
提供通用的辅助函数
"""

import numpy as np
from datetime import datetime
from typing import Any

# 自定义 JSON 编码器函数
def convert_numpy_types(obj):
    """递归转换 NumPy 类型为 Python 原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

def convert_to_serializable(obj: Any) -> Any:
    """
    递归转换对象为 JSON 可序列化的格式
    处理 NumPy 类型、datetime 等

    Args:
        obj: 任意 Python 对象

    Returns:
        JSON 可序列化的对象
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    安全除法，避免除零错误

    Args:
        a: 分子
        b: 分母
        default: 分母为零时的默认值

    Returns:
        除法结果
    """
    if b == 0:
        return default
    return a / b


def format_duration(seconds: float) -> str:
    """
    格式化时长

    Args:
        seconds: 秒数

    Returns:
        格式化的时长字符串 (MM:SS)
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def get_file_size_mb(file_path: str) -> float:
    """
    获取文件大小（MB）

    Args:
        file_path: 文件路径

    Returns:
        文件大小（MB）
    """
    import os
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return 0.0


def ensure_dir(directory: str) -> None:
    """
    确保目录存在，如果不存在则创建

    Args:
        directory: 目录路径
    """
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)


def timestamp_to_datetime(timestamp: float) -> str:
    """
    将时间戳转换为格式化的日期时间字符串

    Args:
        timestamp: Unix 时间戳

    Returns:
        格式化的日期时间字符串
    """
    return datetime.fromtimestamp(timestamp).isoformat()