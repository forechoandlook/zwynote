
[TOC]


## 查对外ip

```sh
curl ifconfig.me
```


## bitcoind 请求方法

这是一个使用 `curl` 向 Bitcoin Core 节点的 JSON-RPC 接口发送的 POST 请求。以下是这个请求的组成部分和常用参数：

### 请求组成部分：
1. `curl` - 命令行工具，用于传输数据
2. `-X POST` - 指定使用 POST 方法
3. `127.0.0.1:18443` - Bitcoin Core 节点的 RPC 接口地址和端口（18443 是测试网的默认端口）
4. `--user xxx:xxxx` - RPC 用户名和密码认证
5. `--data` - 后面跟着要发送的 JSON-RPC 数据
6. `-H "Content-Type:application/json"` - 设置请求头，指定内容类型为 JSON

### 常用 JSON-RPC 参数：
```json
{
  "jsonrpc": "2.0",  // JSON-RPC 版本
  "method": "getmempoolinfo",  // 调用的方法
  "params": [],  // 方法参数（空数组表示无参数）
  "id": 83  // 请求ID（任意数字，用于匹配响应）
}
```

### 常用的 Bitcoin Core RPC 方法（替换 `getmempoolinfo`）：
1. **区块链信息**:
   - `getblockchaininfo` - 获取区块链基本信息
   - `getblockcount` - 获取当前区块高度
   - `getblockhash [height]` - 通过高度获取区块哈希
   - `getblock "hash"` - 获取区块详细信息

2. **内存池信息**:
   - `getmempoolinfo` - 获取内存池信息
   - `getrawmempool` - 获取内存池中所有交易ID

3. **交易相关**:
   - `getrawtransaction "txid"` - 获取原始交易数据
   - `sendrawtransaction "hex"` - 广播原始交易
   - `estimatesmartfee nblocks` - 估算交易费用

4. **钱包相关**:
   - `getbalance` - 获取钱包余额
   - `listunspent` - 列出未花费的输出
   - `sendtoaddress "address" amount` - 发送比特币到地址

5. **网络信息**:
   - `getnetworkinfo` - 获取网络信息
   - `getpeerinfo` - 获取节点连接信息

6. **实用命令**:
   - `help` - 获取命令帮助
   - `validateaddress "address"` - 验证地址有效性

### 常用 curl 选项：
- `-v` - 显示详细输出（调试用）
- `--silent` - 静默模式（不显示进度信息）
- `--output FILE` - 将输出保存到文件
- `--cacert FILE` - 指定 CA 证书（用于 HTTPS）

注意：实际使用时需要替换 `xxx:xxxx` 为 Bitcoin Core 配置文件中设置的 `rpcuser` 和 `rpcpassword`，并确保端口与配置一致（主网默认 8332，测试网默认 18332，regtest 默认 18443）。

## demo

以下是一些常用的 Bitcoin Core JSON-RPC 请求的 `--data` 示例，每个方法都提供一个最简单的 demo：

### 1. 获取区块链信息
```bash
--data '{"jsonrpc":"2.0","method":"getblockchaininfo","params":[],"id":1}'
```

### 2. 获取当前区块高度
```bash
--data '{"jsonrpc":"2.0","method":"getblockcount","params":[],"id":2}'
```

### 3. 通过高度获取区块哈希（例如获取高度 100 的区块哈希）
```bash
--data '{"jsonrpc":"2.0","method":"getblockhash","params":[100],"id":3}'
```

### 4. 获取区块详细信息（需要区块哈希）
```bash
--data '{"jsonrpc":"2.0","method":"getblock","params":["000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"],"id":4}'
```

### 5. 获取内存池信息
```bash
--data '{"jsonrpc":"2.0","method":"getmempoolinfo","params":[],"id":5}'
```

### 6. 获取内存池中的交易列表
```bash
--data '{"jsonrpc":"2.0","method":"getrawmempool","params":[],"id":6}'
```

### 7. 获取原始交易数据（需要交易ID）
```bash
--data '{"jsonrpc":"2.0","method":"getrawtransaction","params":["txid_here"],"id":7}'
```

### 8. 发送原始交易（需要原始交易十六进制）
```bash
--data '{"jsonrpc":"2.0","method":"sendrawtransaction","params":["raw_hex_here"],"id":8}'
```

### 9. 获取钱包余额
```bash
--data '{"jsonrpc":"2.0","method":"getbalance","params":[],"id":9}'
```

### 10. 列出未花费的输出
```bash
--data '{"jsonrpc":"2.0","method":"listunspent","params":[],"id":10}'
```

### 11. 发送比特币到地址（需要地址和金额）
```bash
--data '{"jsonrpc":"2.0","method":"sendtoaddress","params":["tb1qaddress...", 0.01],"id":11}'
```

### 12. 估算交易费用（例如估算 6 个区块确认的费用）
```bash
--data '{"jsonrpc":"2.0","method":"estimatesmartfee","params":[6],"id":12}'
```

### 13. 验证地址有效性
```bash
--data '{"jsonrpc":"2.0","method":"validateaddress","params":["tb1qaddress..."],"id":13}'
```

### 14. 获取网络信息
```bash
--data '{"jsonrpc":"2.0","method":"getnetworkinfo","params":[],"id":14}'
```

### 15. 获取帮助信息
```bash
--data '{"jsonrpc":"2.0","method":"help","params":[],"id":15}'
```

### 完整 curl 示例（以 `getblockcount` 为例）：
```bash
curl -X POST http://127.0.0.1:18443 \
  --user your_rpcuser:your_rpcpassword \
  --data '{"jsonrpc":"2.0","method":"getblockcount","params":[],"id":1}' \
  -H "Content-Type: application/json"
```

注意：
1. 替换 `your_rpcuser:your_rpcpassword` 为你的 Bitcoin Core 配置文件中的 `rpcuser` 和 `rpcpassword`
2. 替换 `127.0.0.1:18443` 为你的节点地址和 RPC 端口（主网默认 8332，测试网 18332，regtest 18443）
3. `txid_here`/`raw_hex_here`/`tb1qaddress...` 等需要替换为实际值

## 总结
123