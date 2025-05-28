import json
import os
from typing import List, Optional
from dataclasses import asdict
from .models import Memory
from config import Config

class MemoryDatabase:
    """记忆数据库 - JSON文件存储实现"""
    
    def __init__(self, config: Config, db_path: str = "memories.json"):
        self.config = config
        self.db_path = db_path
        self.memories = {}
        self._load_memories()
    
    def _load_memories(self):
        """从文件加载记忆"""
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for mem_data in data:
                        memory = Memory(**mem_data)
                        self.memories[memory.id] = memory
        except Exception as e:
            print(f"加载记忆错误: {e}")
    
    def save_memory(self, memory: Memory):
        """保存记忆"""
        self.memories[memory.id] = memory
        self._save_to_file()
    
    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """获取单个记忆"""
        return self.memories.get(memory_id)
    
    def get_memories(self, user_id: str, memory_type: str = None, 
                    session_id: str = None, limit: int = None) -> List[Memory]:
        """获取记忆列表"""
        if limit is None:
            limit = self.config.default_search_limit
        
        filtered_memories = []
        for memory in self.memories.values():
            if memory.user_id != user_id:
                continue
            if memory_type and memory.memory_type != memory_type:
                continue
            if session_id and memory.session_id != session_id:
                continue
            filtered_memories.append(memory)
        
        # 按更新时间排序
        filtered_memories.sort(key=lambda x: x.updated_at, reverse=True)
        return filtered_memories[:limit]
    
    def _save_to_file(self):
        """保存到文件"""
        try:
            data = [asdict(memory) for memory in self.memories.values()]
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存记忆错误: {e}")