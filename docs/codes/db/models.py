from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class MediaType(Enum):
    """媒体类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"  # TODO: 支持PDF、Word等文档

@dataclass
class Memory:
    """记忆数据结构"""
    id: str
    content: str
    memory_type: str  # user, session, entity
    user_id: str
    session_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict] = None
    created_at: str = None
    updated_at: str = None
    
    # 多媒体支持字段
    media_type: MediaType = MediaType.TEXT
    file_path: Optional[str] = None  # 文件存储路径
    file_size: Optional[int] = None  # 文件大小
    duration: Optional[float] = None  # 音频/视频时长
    dimensions: Optional[Dict] = None  # 图片/视频尺寸 {"width": 1920, "height": 1080}
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = self.created_at

@dataclass
class MediaMetadata:
    """多媒体元数据 - TODO
    
    功能规划：
    1. 图片元数据：EXIF信息、拍摄时间、地理位置
    2. 音频元数据：采样率、比特率、编码格式
    3. 视频元数据：帧率、分辨率、编码格式
    """
    file_format: str
    encoding: Optional[str] = None
    quality: Optional[str] = None
    creation_time: Optional[str] = None
    location: Optional[Dict] = None  # GPS坐标
    device_info: Optional[Dict] = None  # 设备信息