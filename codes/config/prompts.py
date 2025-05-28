class PromptTemplates:
    """Prompt模板类"""
    
    MEMORY_EXTRACTION_PROMPT = """
从以下对话中提取重要的记忆信息，返回JSON格式的列表。
每个记忆应该包含：
- content: 记忆内容（简洁明确）
- memory_type: 类型（user/entity/preference）
- importance: 重要性评分（1-10）

对话文本：
{text}

只返回JSON格式，不要其他解释：
"""
    
    CONTEXT_SUMMARY_PROMPT = """
基于以下相关记忆，为用户查询提供简洁的上下文总结：

查询：{query}

相关记忆：
{memories}

请提供一个简洁的上下文总结：
"""
    
    @classmethod
    def get_memory_extraction_prompt(cls, text: str) -> str:
        """获取记忆提取prompt"""
        return cls.MEMORY_EXTRACTION_PROMPT.format(text=text)
    
    @classmethod
    def get_context_summary_prompt(cls, query: str, memories: str) -> str:
        """获取上下文总结prompt"""
        return cls.CONTEXT_SUMMARY_PROMPT.format(query=query, memories=memories)