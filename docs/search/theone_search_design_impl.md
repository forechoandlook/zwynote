# 常驻服务模式实现 (Python版本)

下面是一个使用Python实现的常驻服务方案，结合了SQLite、Xapian和Flask web框架。这个方案适合中小规模的文档检索和查重系统。

## 核心服务架构

```
┌──────────────────────────────────────────────┐
│                   Document Service           │
├──────────────────────────────────────────────┤
│  • Flask HTTP API 接口                        │
│  • SQLite 连接池                              │
│  • Xapian 数据库连接                           │
│  • MinHash 查重功能                            │
└──────────────────────────────────────────────┘
```

## 完整实现代码

### 1. 服务主程序 (`document_service.py`)

```python
import sqlite3
from flask import Flask, request, jsonify
from xapian import Database, Document, TermGenerator, QueryParser
import leveldb
import mmh3
from dataclasses import dataclass
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# 配置常量
SQLITE_DB_PATH = "metadata.db"
XAPIAN_DB_PATH = "xapian_index"
MINHASH_PERMUTATIONS = 128  # MinHash签名长度

app = Flask(__name__)

# 数据库连接池
class ConnectionPool:
    def __init__(self, max_connections=5):
        self.max_connections = max_connections
        self._pool = []
        self._lock = threading.Lock()

    def get_conn(self):
        with self._lock:
            if self._pool:
                return self._pool.pop()
            if len(self._pool) < self.max_connections:
                return sqlite3.connect(SQLITE_DB_PATH)
            raise Exception("Connection pool exhausted")

    def return_conn(self, conn):
        with self._lock:
            self._pool.append(conn)

sqlite_pool = ConnectionPool()
xapian_db = Database(XAPIAN_DB_PATH)

# MinHash 工具函数
def generate_minhash(text: str, num_perm=MINHASH_PERMUTATIONS) -> List[int]:
    shingles = set()
    words = text.split()
    for i in range(len(words) - 2):
        shingles.add(" ".join(words[i:i+3]))

    minhash = [float('inf')] * num_perm
    for shingle in shingles:
        for i in range(num_perm):
            hash_val = mmh3.hash(f"{i}{shingle}", signed=False)
            if hash_val < minhash[i]:
                minhash[i] = hash_val
    return minhash

# API端点
@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '')
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', 10))

    # 1. 使用Xapian进行全文检索
    query_parser = QueryParser()
    query_parser.set_stemming(True)
    xapian_query = query_parser.parse_query(query)

    enquire = xapian_db.enquire(xapian_query)
    matches = enquire.get_mset((page-1)*page_size, page_size)

    # 2. 从SQLite获取元数据
    conn = sqlite_pool.get_conn()
    try:
        cursor = conn.cursor()
        doc_ids = [str(match.docid) for match in matches]
        placeholders = ','.join(['?']*len(doc_ids))

        cursor.execute(f"""
            SELECT id, title, file_path, created_time
            FROM documents
            WHERE id IN ({placeholders})
        """, doc_ids)

        meta_map = {row[0]: row[1:] for row in cursor.fetchall()}

        # 3. 合并结果
        results = []
        for match in matches:
            doc_id = str(match.docid)
            if doc_id in meta_map:
                title, path, created = meta_map[doc_id]
                results.append({
                    'id': doc_id,
                    'title': title,
                    'path': path,
                    'score': match.percent / 100,
                    'created': created
                })

        return jsonify({'results': results})
    finally:
        sqlite_pool.return_conn(conn)

@app.route('/find_similar', methods=['POST'])
def find_similar():
    content = request.json.get('content', '')
    threshold = float(request.json.get('threshold', 0.8))

    # 1. 生成查询MinHash
    query_minhash = generate_minhash(content)

    # 2. 遍历Xapian数据库比较相似度
    results = []
    for doc in xapian_db:
        stored_minhash = [int(x) for x in doc.get_value(0).split(',')]

        # 计算Jaccard相似度估计
        matches = sum(1 for a, b in zip(query_minhash, stored_minhash) if a == b)
        similarity = matches / MINHASH_PERMUTATIONS

        if similarity >= threshold:
            doc_id = doc.get_data()
            conn = sqlite_pool.get_conn()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT title, file_path FROM documents WHERE id = ?
                """, (doc_id,))
                row = cursor.fetchone()
                if row:
                    results.append({
                        'id': doc_id,
                        'title': row[0],
                        'path': row[1],
                        'similarity': similarity
                    })
            finally:
                sqlite_pool.return_conn(conn)

    # 按相似度排序
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return jsonify({'results': results})

# 初始化函数
def initialize_databases():
    # 初始化SQLite
    conn = sqlite3.connect(SQLITE_DB_PATH)
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT,
                file_path TEXT UNIQUE,
                file_type TEXT,
                created_time INTEGER,
                updated_time INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(file_type);
            CREATE INDEX IF NOT EXISTS idx_documents_time ON documents(created_time);
        """)
        conn.commit()
    finally:
        conn.close()

    # 确保Xapian数据库存在
    try:
        Database(XAPIAN_DB_PATH)
    except:
        # 如果不存在则创建
        db = Database(XAPIAN_DB_PATH, Database.CREATE_OR_OPEN)
        db.close()

# 后台索引任务
def background_indexer():
    while True:
        # 这里可以添加定期索引新文档的逻辑
        time.sleep(60 * 5)  # 每5分钟检查一次

if __name__ == '__main__':
    initialize_databases()

    # 启动后台索引线程
    indexer_thread = threading.Thread(target=background_indexer, daemon=True)
    indexer_thread.start()

    # 启动Flask服务
    app.run(host='0.0.0.0', port=5000, threaded=True)
```

### 2. 客户端调用示例

```python
import requests

# 搜索文档
response = requests.get('http://localhost:5000/search', params={
    'q': '人工智能',
    'page': 1,
    'page_size': 10
})
print(response.json())

# 查找相似文档
response = requests.post('http://localhost:5000/find_similar', json={
    'content': '这里是文档的完整内容...',
    'threshold': 0.7
})
print(response.json())
```

## 关键特性

1. **服务化架构**：
   - 基于Flask的HTTP API服务
   - 支持多线程并发请求
   - 常驻内存运行

2. **数据库管理**：
   - SQLite连接池管理
   - Xapian持久化连接
   - 自动初始化数据库

3. **核心功能**：
   - 全文检索 (Xapian)
   - 文档查重 (MinHash)
   - 分页结果返回

4. **扩展能力**：
   - 后台自动索引线程
   - 准备添加新文档的接口

## 部署建议

1. **生产环境部署**：
   ```bash
   # 使用Gunicorn作为WSGI服务器
   gunicorn -w 4 -b 0.0.0.0:5000 document_service:app
   ```

2. **监控和运维**：
   - 添加`/health`端点用于健康检查
   - 使用Supervisor管理进程
   - 记录查询日志

3. **性能优化**：
   - 对高频查询添加缓存
   - 考虑对Xapian数据库分片
   - 对MinHash比较使用更高效的数据结构

这个实现提供了完整的服务框架，您可以根据实际需求进一步扩展功能，如添加用户认证、更复杂的查询参数、或者集成其他存储后端。