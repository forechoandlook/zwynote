#!/usr/bin/env python3
"""
完整的记忆管理系统演示
包含文本记忆和多媒体记忆的使用示例
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from manager import MemoryManager

def demo_text_memory():
    """演示文本记忆功能"""
    print("=== 文本记忆演示 ===")
    
    config = Config(
        openai_api_key=os.environ.get("OPENAI_API_KEY", "test-key"),
        openai_base_url="https://api.zchat.tech/v1"
    )
    
    memory_manager = MemoryManager(config)
    user_id = "demo_user"
    session_id = "demo_session"
    
    # 添加记忆
    conversations = [
        "我喜欢喝咖啡，特别是拿铁",
        "我是一名软件工程师，专注于Python开发",
        "我住在北京，工作在朝阳区",
        "我的生日是3月15日"
    ]
    
    for conv in conversations:
        print(f"添加记忆: {conv}")
        memory_manager.add_memory(conv, user_id, session_id)
    
    # 搜索记忆
    queries = ["咖啡相关", "工作信息", "个人信息"]
    for query in queries:
        print(f"\n查询: {query}")
        context = memory_manager.get_context_for_query(query, user_id)
        print(f"结果: {context}")
    
    memory_manager.cleanup()

def demo_multimedia_memory():
    """演示多媒体记忆功能（TODO）"""
    print("\n=== 多媒体记忆演示 ===")
    print("TODO: 实现图片、音频、视频记忆的演示")
    
    # TODO: 添加图片记忆示例
    # TODO: 添加音频记忆示例
    # TODO: 添加视频记忆示例

if __name__ == "__main__":
    demo_text_memory()
    demo_multimedia_memory()