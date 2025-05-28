from typing import List, Dict
from config import Config

class ImportanceFilter:
    """重要性过滤器"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def filter_by_importance(self, memories_data: List[Dict]) -> List[Dict]:
        """根据重要性过滤记忆"""
        filtered = []
        for mem_data in memories_data:
            importance = mem_data.get('importance', 0)
            if importance >= self.config.min_importance_threshold:
                filtered.append(mem_data)
        return filtered
    
    def deduplicate_memories(self, memories_data: List[Dict]) -> List[Dict]:
        """去重记忆"""
        seen_contents = set()
        deduplicated = []
        
        for mem_data in memories_data:
            content = mem_data.get('content', '')
            if content not in seen_contents:
                seen_contents.add(content)
                deduplicated.append(mem_data)
        
        return deduplicated