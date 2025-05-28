import hashlib
import openai
from typing import List
from codes.config import Config
from codes.db.models import Memory
from codes.db.database import MemoryDatabase
from codes.extract.extractor import MemoryExtractor
from codes.search.vector_store import FaissVectorStore
from codes.filter.importance_filter import ImportanceFilter

class MemoryManager:
    """记忆管理器 - 核心管理类"""
    
    def __init__(self, config: Config):
        self.config = config
        # 初始化组件
        breakpoint()
        self.llm_client = openai.OpenAI(
            api_key=config.openai_api_key, 
            base_url=config.openai_base_url
        )
        self.extractor = MemoryExtractor(self.llm_client, config)
        self.vector_store = FaissVectorStore(config)
        self.database = MemoryDatabase(config)
        self.importance_filter = ImportanceFilter(config)
    
    def generate_memory_id(self, content: str, user_id: str) -> str:
        """生成记忆ID"""
        data = f"{content}_{user_id}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def add_memory(self, text: str, user_id: str, session_id: str = None):
        """添加记忆"""
        # 提取记忆
        extracted_memories = self.extractor.extract_memories(text, user_id, session_id)
        
        # 过滤重要性和去重
        filtered_memories = self.importance_filter.filter_by_importance(extracted_memories)
        filtered_memories = self.importance_filter.deduplicate_memories(filtered_memories)
        
        for mem_data in filtered_memories:
            content = mem_data['content']
            memory_id = self.generate_memory_id(content, user_id)
            
            # 向量化
            embedding = self.vector_store.encode(content)
            
            # 创建记忆对象
            memory = Memory(
                id=memory_id,
                content=content,
                memory_type=mem_data.get('memory_type', 'user'),
                user_id=user_id,
                session_id=session_id,
                embedding=embedding,
                metadata={'importance': mem_data.get('importance', 0)}
            )
            
            # 保存到数据库和向量索引
            self.database.save_memory(memory)
            self.vector_store.add_vector(
                memory_id, 
                embedding, 
                {'user_id': user_id, 'memory_type': memory.memory_type}
            )
            
            print(f"保存记忆: {content}")
        
        # 保存向量索引
        self.vector_store.save_index()
    
    def search_memories(self, query: str, user_id: str, top_k: int = None) -> List[Memory]:
        """搜索相关记忆"""
        if top_k is None:
            top_k = self.config.default_top_k
        
        # 向量化查询
        query_embedding = self.vector_store.encode(query)
        
        # 向量搜索
        search_results = self.vector_store.search(query_embedding, top_k * 2)
        
        # 过滤用户记忆并获取完整记忆对象
        memories = []
        for memory_id, score in search_results:
            memory = self.database.get_memory(memory_id)
            if memory and memory.user_id == user_id:
                memories.append(memory)
                if len(memories) >= top_k:
                    break
        
        return memories
    
    def get_context_for_query(self, query: str, user_id: str, session_id: str = None) -> str:
        """为查询获取上下文记忆"""
        relevant_memories = self.search_memories(query, user_id)
        
        if not relevant_memories:
            return "没有找到相关记忆。"
        
        context_parts = []
        for memory in relevant_memories:
            context_parts.append(f"- {memory.content}")
        
        return "相关记忆：\n" + "\n".join(context_parts)
    
    # TODO: 多媒体记忆功能
    def add_image_memory(self, image_path: str, user_id: str, session_id: str = None):
        """添加图片记忆 - TODO
        
        功能规划：
        1. 图片内容识别和描述生成
        2. 图片特征提取和向量化
        3. OCR文字提取
        4. 图片分类和标签
        5. 图片存储路径管理
        """
        # TODO: 实现图片记忆功能
        pass
    
    def add_audio_memory(self, audio_path: str, user_id: str, session_id: str = None):
        """添加音频记忆 - TODO
        
        功能规划：
        1. 语音转文字(ASR)
        2. 音频特征提取
        3. 说话人识别
        4. 情感分析
        5. 音频文件存储管理
        """
        # TODO: 实现音频记忆功能
        pass
    
    def add_video_memory(self, video_path: str, user_id: str, session_id: str = None):
        """添加视频记忆 - TODO
        
        功能规划：
        1. 视频关键帧提取
        2. 视频内容描述生成
        3. 音频轨道处理
        4. 视频场景分割
        5. 视频文件存储管理
        """
        # TODO: 实现视频记忆功能
        pass
    
    def search_multimedia_memories(self, query: str, media_type: str, user_id: str) -> List[Memory]:
        """搜索多媒体记忆 - TODO
        
        功能规划：
        1. 跨模态搜索(文本搜图片/音频/视频)
        2. 相似图片搜索
        3. 音频相似度搜索
        4. 视频内容搜索
        """
        # TODO: 实现多媒体搜索功能
        pass
    
    def cleanup(self):
        """清理资源"""
        self.vector_store.save_index()