import json
import os
from typing import List, Dict, Tuple
import numpy as np
import ollama
import faiss
# 修改为相对导入
from ..config import Config

class FaissVectorStore:
    """Faiss向量存储 - 处理向量化和相似性搜索"""
    
    def __init__(self, config: Config):
        self.config = config
        self.dimension = config.vector_dimension
        self.index = None
        self.metadata = []
        self.id_to_index = {}
        self._init_index()
    
    def _init_index(self):
        """初始化Faiss索引"""
        self.index = faiss.IndexFlatIP(self.dimension)  # 使用内积相似度
        self._load_index()
    
    def encode(self, text: str) -> List[float]:
        """文本向量化，使用Ollama API"""
        try:
            embedding = ollama.embed(model=self.config.ollama_model, input=text)
            embedding = embedding.embeddings[0]
            # 标准化向量以便使用内积计算余弦相似度
            embedding = np.array(embedding, dtype=np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            return embedding.tolist()
        except Exception as e:
            print(f"向量化错误: {e}")
            return [0.0] * self.dimension
    
    def add_vector(self, memory_id: str, embedding: List[float], metadata: Dict):
        """添加向量到索引"""
        embedding_array = np.array([embedding], dtype=np.float32)
        self.index.add(embedding_array)
        
        # 更新元数据
        index_id = len(self.metadata)
        self.metadata.append({
            'memory_id': memory_id,
            'metadata': metadata
        })
        self.id_to_index[memory_id] = index_id
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """搜索相似向量"""
        if self.index.ntotal == 0:
            return []
        
        query_array = np.array([query_embedding], dtype=np.float32)
        scores, indices = self.index.search(query_array, min(top_k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # 有效索引
                memory_id = self.metadata[idx]['memory_id']
                results.append((memory_id, float(score)))
        
        return results
    
    def save_index(self):
        """保存索引到文件"""
        try:
            faiss.write_index(self.index, self.config.faiss_index_path)
            with open(self.config.faiss_metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': self.metadata,
                    'id_to_index': self.id_to_index
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存索引错误: {e}")
    
    def _load_index(self):
        """从文件加载索引"""
        try:
            if os.path.exists(self.config.faiss_index_path):
                self.index = faiss.read_index(self.config.faiss_index_path)
            
            if os.path.exists(self.config.faiss_metadata_path):
                with open(self.config.faiss_metadata_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.metadata = data.get('metadata', [])
                    self.id_to_index = data.get('id_to_index', {})
        except Exception as e:
            print(f"加载索引错误: {e}")
            self._init_index()