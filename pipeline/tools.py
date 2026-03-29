import numpy as np

def convert_to_serializable(obj):
    """
    递归转换任意对象为 JSON 可序列化的格式
    处理 numpy 类型、自定义对象等
    """
    # 处理 None
    if obj is None:
        return None

    # 处理基本类型（已经可序列化）
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # 处理 numpy 类型
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)

    # 处理字典
    if isinstance(obj, dict):
        return {convert_to_serializable(key): convert_to_serializable(value)
                for key, value in obj.items()}

    # 处理列表、元组、集合
    if isinstance(obj, (list, tuple, set)):
        return [convert_to_serializable(item) for item in obj]

    # 处理自定义对象（如果有 __dict__ 属性）
    if hasattr(obj, '__dict__'):
        return convert_to_serializable(obj.__dict__)

    # 其他类型转为字符串
    try:
        return str(obj)
    except:
        return f"<unserializable: {type(obj).__name__}>"