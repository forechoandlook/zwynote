# 使用指南

## 本地开发

### 环境准备

1. 安装 Python 3.x
2. 安装 MkDocs 和 Material 主题：

```bash
pip install mkdocs-material
```

### 本地运行

1. 克隆仓库：

```bash
git clone <repository-url>
cd my-notes
```

2. 启动本地服务器：

```bash
mkdocs serve
```

3. 在浏览器中访问 `http://127.0.0.1:8000` 查看文档

## 文档结构

```
docs/
├── index.md      # 首页
├── guide.md      # 使用指南
└── cc/           # C++ 相关文档
    └── mix.md    # Mix 模板编程
```

## 贡献指南

### 添加新文档

1. 在 `docs` 目录下创建 Markdown 文件
2. 在 `mkdocs.yml` 的 `nav` 部分添加文档链接
3. 提交 Pull Request

### Markdown 规范

- 使用 ATX 风格标题（`#` 号）
- 代码块指定语言类型
- 适当使用表格、列表等 Markdown 元素
- 保持文档结构清晰

### 本地预览

修改文档后，本地服务器会自动重新加载，实时预览更改。

## 部署

本文档使用 GitHub Actions 自动部署到 GitHub Pages。每次推送到 main 分支时会自动触发部署流程。