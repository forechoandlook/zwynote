import os
from config import Config, MediaType
from manager import MemoryManager

def multimedia_demo():
    """多媒体记忆示例 - TODO"""
    config = Config(
        openai_api_key=os.environ.get("OPENAI_API_KEY", "test-key")
    )
    
    memory_manager = MemoryManager(config)
    user_id = "multimedia_user"
    
    print("多媒体记忆功能演示 (TODO)")
    
    # TODO: 图片记忆示例
    print("\n1. 图片记忆功能:")
    image_path = "example_image.jpg"
    if config.is_supported_format(image_path, MediaType.IMAGE):
        print(f"支持的图片格式: {image_path}")
        # memory_manager.add_image_memory(image_path, user_id)
    
    # TODO: 音频记忆示例
    print("\n2. 音频记忆功能:")
    audio_path = "example_audio.mp3"
    if config.is_supported_format(audio_path, MediaType.AUDIO):
        print(f"支持的音频格式: {audio_path}")
        # memory_manager.add_audio_memory(audio_path, user_id)
    
    # TODO: 视频记忆示例
    print("\n3. 视频记忆功能:")
    video_path = "example_video.mp4"
    if config.is_supported_format(video_path, MediaType.VIDEO):
        print(f"支持的视频格式: {video_path}")
        # memory_manager.add_video_memory(video_path, user_id)
    
    # TODO: 跨模态搜索示例
    print("\n4. 跨模态搜索功能:")
    # results = memory_manager.search_multimedia_memories("找到包含猫的图片", "image", user_id)
    
    memory_manager.cleanup()

if __name__ == "__main__":
    multimedia_demo()