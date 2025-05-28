import os
from dataclasses import dataclass
from typing import Optional
from enum import Enum

class MediaType(Enum):
    """媒体类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"

@dataclass
class Config:
    """配置类"""
    # OpenAI配置
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.zchat.tech/v1")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    # Ollama配置
    ollama_model: str = os.getenv("OLLAMA_MODEL", "nomic-embed-text:latest")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # 向量配置
    vector_dimension: int = 768
    
    # 记忆配置
    min_importance_threshold: int = 5
    default_search_limit: int = 100
    default_top_k: int = 5
    
    # Faiss配置
    faiss_index_path: str = "memory_index.faiss"
    faiss_metadata_path: str = "memory_metadata.json"
    
    # 温度参数
    llm_temperature: float = 0.1
    
    # 文件存储配置
    media_storage_path: str = "media_storage/"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    
    # 图片处理配置
    image_max_width: int = 1920
    image_max_height: int = 1080
    image_quality: int = 85
    supported_image_formats: list = None
    
    # 音频处理配置
    audio_sample_rate: int = 16000
    audio_max_duration: int = 3600  # 1小时
    supported_audio_formats: list = None
    
    # 视频处理配置
    video_max_duration: int = 7200  # 2小时
    video_frame_extract_interval: int = 30  # 每30秒提取一帧
    supported_video_formats: list = None
    
    # ASR配置 - TODO
    whisper_model: str = "base"
    whisper_language: str = "zh"
    
    # 图像识别配置 - TODO
    vision_model: str = "gpt-4-vision-preview"
    
    # 数据库配置
    database_path: str = "memories.json"
    backup_enabled: bool = True
    backup_interval: int = 3600  # 1小时备份一次
    
    # 日志配置
    log_level: str = "INFO"
    log_file: str = "memory_system.log"
    
    # 性能配置
    batch_size: int = 32
    max_concurrent_requests: int = 10
    cache_size: int = 1000
    
    def __post_init__(self):
        if self.supported_image_formats is None:
            self.supported_image_formats = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        if self.supported_audio_formats is None:
            self.supported_audio_formats = ['.mp3', '.wav', '.flac', '.aac', '.ogg']
        if self.supported_video_formats is None:
            self.supported_video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    
    def validate(self) -> bool:
        """验证配置是否有效"""
        if not self.openai_api_key:
            print("警告: OpenAI API Key 未设置")
            return False
        
        if self.vector_dimension <= 0:
            print("错误: 向量维度必须大于0")
            return False
        
        if self.min_importance_threshold < 1 or self.min_importance_threshold > 10:
            print("错误: 重要性阈值必须在1-10之间")
            return False
        
        return True
    
    def get_media_storage_path(self, media_type: MediaType) -> str:
        """获取特定媒体类型的存储路径"""
        return os.path.join(self.media_storage_path, media_type.value)
    
    def is_supported_format(self, file_path: str, media_type: MediaType) -> bool:
        """检查文件格式是否支持"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if media_type == MediaType.IMAGE:
            return ext in self.supported_image_formats
        elif media_type == MediaType.AUDIO:
            return ext in self.supported_audio_formats
        elif media_type == MediaType.VIDEO:
            return ext in self.supported_video_formats
        
        return False