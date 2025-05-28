import os
# 修改为相对导入
from ..config import Config
from ..manager import MemoryManager

def main():
    """基本使用示例"""
    # 设置环境变量（实际使用时）
    # os.environ["OPENAI_API_KEY"] = "your-api-key"
    
    # 创建配置
    config = Config(
        openai_api_key=os.environ.get("OPENAI_API_KEY", "test-key"),
        openai_base_url="https://api.zchat.tech/v1"
    )
    
    # 验证配置
    if not config.validate():
        print("配置验证失败，请检查配置")
        return
    
    # 初始化记忆管理器
    memory_manager = MemoryManager(config)
    
    # 模拟对话和记忆添加
    user_id = "user_001"
    session_id = "session_001"
    
    # 添加记忆
    conversations = [
        "我喜欢喝咖啡，特别是拿铁",
        "我是一名软件工程师，专注于Python开发",
        "我住在北京，工作在朝阳区",
        "我的生日是3月15日"
    ]
    
    print("添加记忆...")
    for conv in conversations:
        print(f"处理对话: {conv}")
        memory_manager.add_memory(conv, user_id, session_id)
        print("---")
    
    # 搜索记忆
    print("\n搜索测试:")
    queries = ["咖啡相关", "工作信息", "个人信息"]
    
    for query in queries:
        print(f"\n查询: {query}")
        context = memory_manager.get_context_for_query(query, user_id)
        print(context)
    
    # 清理资源
    memory_manager.cleanup()
    print("\n记忆管理器已清理")

if __name__ == "__main__":
    main()