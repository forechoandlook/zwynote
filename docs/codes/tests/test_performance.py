import unittest
import time
import tempfile
import shutil
from config import Config
from db import MemoryDatabase, Memory
from search import FaissVectorStore

class TestPerformance(unittest.TestCase):
    """性能测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp()
        self.config = Config(
            faiss_index_path=os.path.join(self.test_dir, "perf_index.faiss"),
            faiss_metadata_path=os.path.join(self.test_dir, "perf_metadata.json"),
            database_path=os.path.join(self.test_dir, "perf_memories.json")
        )
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.test_dir)
    
    def test_database_performance(self):
        """测试数据库性能"""
        db = MemoryDatabase(self.config)
        
        # 测试批量插入性能
        start_time = time.time()
        
        for i in range(1000):
            memory = Memory(
                id=f"perf_test_{i}",
                content=f"性能测试记忆 {i}",
                memory_type="user",
                user_id="perf_user"
            )
            db.save_memory(memory)
        
        insert_time = time.time() - start_time
        print(f"插入1000条记忆耗时: {insert_time:.2f}秒")
        
        # 测试查询性能
        start_time = time.time()
        
        for i in range(100):
            memory = db.get_memory(f"perf_test_{i}")
            self.assertIsNotNone(memory)
        
        query_time = time.time() - start_time
        print(f"查询100条记忆耗时: {query_time:.2f}秒")
        
        # 性能断言
        self.assertLess(insert_time, 10.0, "插入性能不达标")
        self.assertLess(query_time, 1.0, "查询性能不达标")
    
    def test_vector_search_performance(self):
        """测试向量搜索性能 - TODO"""
        # TODO: 实现向量搜索性能测试
        pass
    
    def test_memory_usage(self):
        """测试内存使用 - TODO"""
        # TODO: 实现内存使用测试
        pass

if __name__ == "__main__":
    unittest.main()