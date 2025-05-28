import unittest
import os
import tempfile
import shutil
from config import Config
from manager import MemoryManager
from db import Memory, MemoryDatabase
from search import FaissVectorStore

class TestMemorySystem(unittest.TestCase):
    """记忆系统测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时目录
        self.test_dir = tempfile.mkdtemp()
        
        # 创建测试配置
        self.config = Config(
            openai_api_key="test-key",
            faiss_index_path=os.path.join(self.test_dir, "test_index.faiss"),
            faiss_metadata_path=os.path.join(self.test_dir, "test_metadata.json"),
            database_path=os.path.join(self.test_dir, "test_memories.json")
        )
        
        # 模拟向量存储（不依赖实际API）
        self.vector_store = MockVectorStore(self.config)
        
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.test_dir)
    
    def test_memory_creation(self):
        """测试记忆创建"""
        memory = Memory(
            id="test_id",
            content="测试内容",
            memory_type="user",
            user_id="user_001"
        )
        
        self.assertEqual(memory.id, "test_id")
        self.assertEqual(memory.content, "测试内容")
        self.assertEqual(memory.user_id, "user_001")
        self.assertIsNotNone(memory.created_at)
    
    def test_memory_database(self):
        """测试记忆数据库"""
        db = MemoryDatabase(self.config)
        
        # 创建测试记忆
        memory = Memory(
            id="test_id",
            content="测试记忆内容",
            memory_type="user",
            user_id="user_001"
        )
        
        # 保存记忆
        db.save_memory(memory)
        
        # 获取记忆
        retrieved_memory = db.get_memory("test_id")
        self.assertIsNotNone(retrieved_memory)
        self.assertEqual(retrieved_memory.content, "测试记忆内容")
        
        # 获取用户记忆列表
        user_memories = db.get_memories("user_001")
        self.assertEqual(len(user_memories), 1)
        self.assertEqual(user_memories[0].id, "test_id")
    
    def test_vector_store(self):
        """测试向量存储"""
        # 添加测试向量
        embedding1 = [0.1, 0.2, 0.3] + [0.0] * (self.config.vector_dimension - 3)
        embedding2 = [0.2, 0.3, 0.4] + [0.0] * (self.config.vector_dimension - 3)
        
        self.vector_store.add_vector("mem1", embedding1, {"type": "test"})
        self.vector_store.add_vector("mem2", embedding2, {"type": "test"})
        
        # 搜索相似向量
        query_embedding = [0.15, 0.25, 0.35] + [0.0] * (self.config.vector_dimension - 3)
        results = self.vector_store.search(query_embedding, top_k=2)
        
        self.assertEqual(len(results), 2)
        self.assertIn("mem1", [r[0] for r in results])
        self.assertIn("mem2", [r[0] for r in results])
    
    def test_config_validation(self):
        """测试配置验证"""
        # 测试有效配置
        valid_config = Config(openai_api_key="test-key")
        self.assertTrue(valid_config.validate())
        
        # 测试无效配置
        invalid_config = Config(openai_api_key="")
        self.assertFalse(invalid_config.validate())
    
    def test_media_type_support(self):
        """测试媒体类型支持"""
        config = Config()
        
        # 测试图片格式
        self.assertTrue(config.is_supported_format("test.jpg", config.MediaType.IMAGE))
        self.assertFalse(config.is_supported_format("test.txt", config.MediaType.IMAGE))
        
        # 测试音频格式
        self.assertTrue(config.is_supported_format("test.mp3", config.MediaType.AUDIO))
        self.assertFalse(config.is_supported_format("test.jpg", config.MediaType.AUDIO))

class MockVectorStore(FaissVectorStore):
    """模拟向量存储，用于测试"""
    
    def encode(self, text: str):
        """模拟向量化"""
        # 简单的文本哈希作为向量
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        
        # 转换为浮点数向量
        vector = []
        for i in range(0, len(hash_hex), 2):
            val = int(hash_hex[i:i+2], 16) / 255.0
            vector.append(val)
        
        # 填充到指定维度
        while len(vector) < self.dimension:
            vector.append(0.0)
        
        return vector[:self.dimension]

def run_functional_tests():
    """运行功能测试"""
    print("开始功能测试...")
    
    # 测试配置加载
    config = Config()
    print(f"✓ 配置加载成功: {config.openai_model}")
    
    # 测试记忆创建
    memory = Memory(
        id="func_test",
        content="功能测试记忆",
        memory_type="user",
        user_id="test_user"
    )
    print(f"✓ 记忆创建成功: {memory.content}")
    
    # 测试数据库操作
    db = MemoryDatabase(config)
    db.save_memory(memory)
    retrieved = db.get_memory("func_test")
    assert retrieved is not None
    print("✓ 数据库操作成功")
    
    # 清理测试文件
    if os.path.exists(config.database_path):
        os.remove(config.database_path)
    
    print("所有功能测试通过！")

if __name__ == "__main__":
    # 运行单元测试
    print("运行单元测试...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "="*50)
    
    # 运行功能测试
    run_functional_tests()