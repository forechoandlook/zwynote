# AMEM

需要解决的问题： 如何为 LLM Agent 设计一个灵活通用的记忆系统，以支持其与外部环境的长期交互？

具体问题：
1. 记忆碎片化
2. 记忆之间的关系，如何更新
3. 预定义模式（如树状结构/图数据库）与开放场景知识演化 

希望：
1. 自主生成，自主生成记忆的上下文描述，动态建立记忆连接，根据新的经验更新记忆
2. 动态组织，动态结构？


```python
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import json
import numpy as np

class AgenticMemorySystem:
    def __init__(self):
        # 初始化文本编码器（使用sentence-transformers的all-MiniLM-L6-v2）
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 记忆库存储结构：列表存储记忆节点字典
        self.memory_bank = []  
        
        # 超参数设置
        self.top_k = 10  # 相似记忆检索数量
        self.sim_threshold = 0.7  # 相似度阈值
        
    class MemoryNode:
        """记忆节点数据结构定义
        $$ m_i = \{c_i, t_i, K_i, G_i, X_i, e_i, L_i\} $$
        """
        def __init__(self, content: str, timestamp: float):
            self.content = content      # 原始交互内容 c_i
            self.timestamp = timestamp  # 时间戳 t_i
            self.keywords = []          # 关键词 K_i
            self.tags = []              # 分类标签 G_i
            self.context = ""           # 上下文描述 X_i
            self.embedding = None       # 嵌入向量 e_i
            self.links = []             # 链接记忆 L_i

    def _generate_note_components(self, content: str) -> Dict:
        """原子笔记构建模块（模拟LLM生成过程）
        论文公式：K_i, G_i, X_i ← LLM(c_i ‖ t_i ‖ P_{s1})
        """
        # 实际应用时替换为真实LLM调用
        return {
            "keywords": ["人工智能", "记忆系统", "神经网络"],
            "tags": ["技术", "机器学习", "认知科学"],
            "context": "该对话涉及人工智能记忆系统的技术原理及其在神经网络中的应用"
        }

    def _generate_links(self, new_node: MemoryNode, candidates: List[MemoryNode]) -> List[int]:
        """动态链接生成模块
        论文公式：L_i ← LLM(m_n ‖ M_{near}^n ‖ P_{s2})
        """
        # 基于语义相似度和规则模拟链接生成
        linked_ids = []
        for candidate in candidates:
            if len(linked_ids) >= 3:  # 最大链接数限制
                break
            if self._semantic_relationship(new_node, candidate):
                linked_ids.append(id(candidate))
        return linked_ids

    def _semantic_relationship(self, node_a: MemoryNode, node_b: MemoryNode) -> bool:
        """语义关系判断（模拟LLM推理过程）"""
        # 实际应用应使用LLM判断，这里用关键词重叠模拟
        common_keywords = set(node_a.keywords) & set(node_b.keywords)
        return len(common_keywords) >= 1

    def _update_memory_evolution(self, new_node: MemoryNode, linked_nodes: List[MemoryNode]):
        """记忆进化更新模块
        论文公式：m_j^* ← LLM(m_n ‖ M_{near}^n \ m_j ‖ m_j ‖ P_{s3})
        """
        for old_node in linked_nodes:
            # 合并关键词（去重）
            combined_keywords = list(set(old_node.keywords + new_node.keywords))
            
            # 更新上下文描述
            old_node.context = f"{old_node.context} | 关联更新：{new_node.context[:50]}…"
            
            # 更新嵌入向量（加权平均）
            old_embed = old_node.embedding
            new_embed = new_node.embedding
            old_node.embedding = (old_embed * 0.7 + new_embed * 0.3) / (0.7 + 0.3)

    def add_memory(self, content: str):
        """添加新记忆的核心流程"""
        # 1. 创建基础记忆节点
        new_node = self.MemoryNode(
            content=content,
            timestamp=time.time()
        )
        
        # 2. 生成结构化属性（模拟LLM过程）
        components = self._generate_note_components(content)
        new_node.keywords = components["keywords"]
        new_node.tags = components["tags"]
        new_node.context = components["context"]
        
        # 3. 生成嵌入向量 e_i = f_enc[concat(c_i, K_i, G_i, X_i)]
        text_to_encode = f"{content} {' '.join(new_node.keywords)} {' '.join(new_node.tags)} {new_node.context}"
        new_node.embedding = self.encoder.encode(text_to_encode, convert_to_tensor=True)
        
        # 4. 相似记忆检索（余弦相似度计算）
        if self.memory_bank:
            embeddings = torch.stack([node.embedding for node in self.memory_bank])
            sim_scores = cosine_similarity(
                new_node.embedding.unsqueeze(0),
                embeddings
            )
            top_k_indices = sim_scores.argsort()[0][-self.top_k:]
            candidates = [self.memory_bank[i] for i in top_k_indices]
        else:
            candidates = []
        
        # 5. 动态链接生成
        new_node.links = self._generate_links(new_node, candidates)
        
        # 6. 记忆进化更新
        if candidates:
            self._update_memory_evolution(new_node, candidates)
        
        # 7. 存入记忆库
        self.memory_bank.append(new_node)

    def retrieve_memories(self, query: str, k: int = 5) -> List[MemoryNode]:
        """记忆检索模块
        论文公式：M_{retrieved} = {m_i | rank(s_{q,i}) ≤ k}
        """
        # 编码查询语句
        query_embed = self.encoder.encode(query, convert_to_tensor=True)
        
        # 计算相似度
        embeddings = torch.stack([node.embedding for node in self.memory_bank])
        sim_scores = cosine_similarity(
            query_embed.unsqueeze(0),
            embeddings
        )
        
        # 排序并返回Top-k结果
        top_k_indices = sim_scores.argsort()[0][-k:]
        return [self.memory_bank[i] for i in top_k_indices]

# 使用示例
if __name__ == "__main__":
    # 初始化记忆系统
    ams = AgenticMemorySystem()
    
    # 添加示例记忆
    ams.add_memory("论文提出了一种基于Zettelkasten的主动记忆系统")
    ams.add_memory("深度学习模型的记忆机制需要动态更新策略")
    
    # 检索相关记忆
    results = ams.retrieve_memories("人工智能记忆系统", k=2)
    for node in results:
        print(f"内容：{node.content[:50]}…")
        print(f"关键词：{node.keywords}")
        print(f"上下文：{node.context[:60]}…\n")
```