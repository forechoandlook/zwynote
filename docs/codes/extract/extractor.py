import json
import re
from typing import List, Dict
from config import Config
from config.prompts import PromptTemplates

class MemoryExtractor:
    """记忆提取器 - 从对话中提取关键信息"""
    
    def __init__(self, llm_client, config: Config):
        self.llm_client = llm_client
        self.config = config
    
    def extract_memories(self, text: str, user_id: str, session_id: str) -> List[Dict]:
        """从文本中提取记忆"""
        prompt = PromptTemplates.get_memory_extraction_prompt(text)
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.llm_temperature
            )
            
            result = response.choices[0].message.content.strip()
            # 提取JSON部分
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            if json_match:
                memories_data = json.loads(json_match.group())
                return memories_data
            else:
                print("未找到有效的JSON格式")
                return []
        except Exception as e:
            print(f"记忆提取错误: {e}")
            return []