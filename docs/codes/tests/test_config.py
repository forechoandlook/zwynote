import unittest
import os
import tempfile
from config import Config, MediaType

class TestConfig(unittest.TestCase):
    """配置测试类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = Config()
        
        # 测试默认值
        self.assertEqual(config.openai_model, "gpt-3.5-turbo")
        self.assertEqual(config.vector_dimension, 768)
        self.assertEqual(config.min_importance_threshold, 5)
        self.assertEqual(config.llm_temperature, 0.1)
    
    def test_environment_variables(self):
        """测试环境变量配置"""
        # 设置环境变量
        os.environ["OPENAI_API_KEY"] = "test-key-123"
        os.environ["OPENAI_MODEL"] = "gpt-4"
        
        config = Config()
        
        self.assertEqual(config.openai_api_key, "test-key-123")
        self.assertEqual(config.openai_model, "gpt-4")
        
        # 清理环境变量
        del os.environ["OPENAI_API_KEY"]
        del os.environ["OPENAI_MODEL"]
    
    def test_config_validation(self):
        """测试配置验证"""
        # 有效配置
        valid_config = Config(openai_api_key="valid-key")
        self.assertTrue(valid_config.validate())
        
        # 无效配置 - 缺少API Key
        invalid_config1 = Config(openai_api_key="")
        self.assertFalse(invalid_config1.validate())
        
        # 无效配置 - 错误的向量维度
        invalid_config2 = Config(openai_api_key="valid-key", vector_dimension=0)
        self.assertFalse(invalid_config2.validate())
        
        # 无效配置 - 错误的重要性阈值
        invalid_config3 = Config(openai_api_key="valid-key", min_importance_threshold=15)
        self.assertFalse(invalid_config3.validate())
    
    def test_media_type_enum(self):
        """测试媒体类型枚举"""
        self.assertEqual(MediaType.TEXT.value, "text")
        self.assertEqual(MediaType.IMAGE.value, "image")
        self.assertEqual(MediaType.AUDIO.value, "audio")
        self.assertEqual(MediaType.VIDEO.value, "video")
        self.assertEqual(MediaType.DOCUMENT.value, "document")
    
    def test_supported_formats(self):
        """测试支持的文件格式"""
        config = Config()
        
        # 测试图片格式
        self.assertTrue(config.is_supported_format("test.jpg", MediaType.IMAGE))
        self.assertTrue(config.is_supported_format("test.PNG", MediaType.IMAGE))  # 大小写不敏感
        self.assertFalse(config.is_supported_format("test.txt", MediaType.IMAGE))
        
        # 测试音频格式
        self.assertTrue(config.is_supported_format("test.mp3", MediaType.AUDIO))
        self.assertTrue(config.is_supported_format("test.WAV", MediaType.AUDIO))
        self.assertFalse(config.is_supported_format("test.jpg", MediaType.AUDIO))
        
        # 测试视频格式
        self.assertTrue(config.is_supported_format("test.mp4", MediaType.VIDEO))
        self.assertTrue(config.is_supported_format("test.AVI", MediaType.VIDEO))
        self.assertFalse(config.is_supported_format("test.mp3", MediaType.VIDEO))
    
    def test_media_storage_path(self):
        """测试媒体存储路径"""
        config = Config(media_storage_path="/tmp/media/")
        
        image_path = config.get_media_storage_path(MediaType.IMAGE)
        audio_path = config.get_media_storage_path(MediaType.AUDIO)
        video_path = config.get_media_storage_path(MediaType.VIDEO)
        
        self.assertEqual(image_path, "/tmp/media/image")
        self.assertEqual(audio_path, "/tmp/media/audio")
        self.assertEqual(video_path, "/tmp/media/video")
    
    def test_custom_config(self):
        """测试自定义配置"""
        custom_config = Config(
            openai_api_key="custom-key",
            vector_dimension=1024,
            min_importance_threshold=7,
            llm_temperature=0.5
        )
        
        self.assertEqual(custom_config.openai_api_key, "custom-key")
        self.assertEqual(custom_config.vector_dimension, 1024)
        self.assertEqual(custom_config.min_importance_threshold, 7)
        self.assertEqual(custom_config.llm_temperature, 0.5)

if __name__ == "__main__":
    unittest.main()