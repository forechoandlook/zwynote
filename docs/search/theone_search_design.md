# 简化版高效存储方案：平衡功能与复杂度

确实之前的方案考虑得较为全面但也相对复杂。下面我提供一个更简洁的实现方案，保留核心功能的同时大幅降低实现复杂度。

## 简化后架构设计

```
┌───────────────────────┐
│      应用层           │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│      Xapian          │  # 主存储
│   • 全文内容          │
│   • 基本元数据        │
│   • MinHash签名       │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│      SQLite          │  # 辅助存储
│   • 系统配置          │
│   • 少量关系数据      │
└──────────────────────┘
```

## 关键简化点

1. **去除LevelDB**：将MinHash直接存储在Xapian的value slot中
2. **减少数据冗余**：只保留必要的元数据副本
3. **简化查询流程**：合并查重和检索的逻辑路径

## 具体实现方案

### 1. 数据结构设计

**Xapian存储**（主存储）：
- 文档内容（可检索）
- 文档标题（可检索，更高权重）
- MinHash签名（存储在value slot）
- 基本元数据（创建时间、作者等）

**SQLite存储**（仅用于）：
- 用户配置
- 系统状态
- 需要复杂查询的关系数据

### 2. 核心代码实现

#### 文档索引

```cpp
void indexDocument(const Document& doc) {
    // 1. 生成MinHash
    vector<uint64_t> minhash = generateMinHash(doc.content);

    // 2. 创建Xapian文档
    Xapian::Document xdoc;
    xdoc.set_data(doc.id); // 只存储ID

    // 添加可检索内容
    Xapian::TermGenerator termgen;
    termgen.set_document(xdoc);
    termgen.index_text(doc.title, 1, "S"); // 标题更高权重
    termgen.index_text(doc.content);

    // 存储元数据为term（便于过滤）
    xdoc.add_term("type_" + doc.type);
    xdoc.add_term("author_" + doc.author);

    // 存储MinHash（压缩后放入value slot）
    xdoc.add_value(MINHASH_SLOT, compressMinHash(minhash));

    // 3. 提交到数据库
    db.replace_document(doc.id, xdoc);
}
```

#### 文档查重

```cpp
vector<DuplicateResult> findDuplicates(string content) {
    // 1. 生成查询MinHash
    auto query_hash = generateMinHash(content);

    // 2. 遍历数据库计算相似度（实际应用应优化此步骤）
    vector<DuplicateResult> results;
    Xapian::PostingIterator it;
    for (it = db.postlist_begin(""); it != db.postlist_end(""); ++it) {
        Xapian::Document doc = db.get_document(*it);
        string stored_hash = doc.get_value(MINHASH_SLOT);

        double similarity = compareHashes(
            query_hash,
            decompressMinHash(stored_hash)
        );

        if (similarity > THRESHOLD) {
            results.push_back({
                doc.get_data(), // 存储的doc.id
                doc.get_value(TITLE_SLOT),
                similarity
            });
        }
    }

    // 3. 按相似度排序
    sort(results.begin(), results.end(), [](auto& a, auto& b) {
        return a.similarity > b.similarity;
    });

    return results;
}
```

#### 混合查询

```cpp
vector<SearchResult> search(const Query& query) {
    if (query.mode == SIMILARITY_SEARCH) {
        return findDuplicates(query.content);
    }

    // 普通全文检索
    Xapian::Enquire enquire(db);
    Xapian::Query xquery = parseQuery(query);
    enquire.set_query(xquery);

    vector<SearchResult> results;
    Xapian::MSet matches = enquire.get_mset(0, 20);
    for (auto it = matches.begin(); it != matches.end(); ++it) {
        Xapian::Document doc = it.get_document();
        results.push_back({
            doc.get_data(),
            doc.get_value(TITLE_SLOT),
            it.get_percent() / 100.0
        });
    }

    return results;
}
```

### 3. 性能优化策略

1. **MinHash存储优化**：
   ```cpp
   string compressMinHash(const vector<uint64_t>& hashes) {
       // 使用简单的差值编码
       stringstream ss;
       uint64_t last = 0;
       for (auto h : hashes) {
           ss << encodeVarint(h - last);
           last = h;
       }
       return ss.str();
   }
   ```

2. **查重优化**（替代全量扫描）：
   ```cpp
   vector<DuplicateResult> optimizedFindDuplicates(string content) {
       auto query_hash = generateMinHash(content);

       // 1. 提取关键band作为过滤条件
       vector<string> query_terms;
       for (int i = 0; i < 3; i++) { // 使用前3个band
           query_terms.push_back(
               fmt::format("band{}:{}", i, query_hash.bands[i]));
       }

       // 2. 构建Xapian查询
       Xapian::Query query(Xapian::Query::OP_OR,
           query_terms.begin(), query_terms.end());

       // 其余逻辑相同...
   }
   ```

3. **缓存热点数据**：
   ```cpp
   class DocumentCache {
   public:
       string getTitle(Xapian::docid id) {
           if (auto it = title_cache.find(id); it != title_cache.end()) {
               return it->second;
           }
           auto doc = db.get_document(id);
           string title = doc.get_value(TITLE_SLOT);
           title_cache[id] = title;
           return title;
       }

   private:
       unordered_map<Xapian::docid, string> title_cache;
       Xapian::Database& db;
   };
   ```

## 方案优势

1. **实现简单**：
   - 主要逻辑都在Xapian中实现
   - SQLite仅用于辅助存储
   - 代码量减少50%以上

2. **维护方便**：
   - 只有两个存储引擎需要管理
   - 备份和恢复流程简单

3. **资源需求低**：
   - 不需要维护LevelDB
   - 内存占用更小

4. **足够应对**：
   - 万级文档规模
   - 中等查询频率
   - 基本查重需求

## 适用场景推荐

这个简化方案特别适合以下情况：
- 文档数量在10万以内
- 每天更新不超过1000次
- 不需要分布式部署
- 开发资源有限的小型项目

如果后期需要扩展，可以从以下方面增强：
1. 将MinHash迁移到Redis等内存数据库
2. 对Xapian数据库进行分片
3. 添加查询结果缓存层

这样在保持系统简单性的同时，也为未来可能的扩展留下了空间。