# 智能文档问答系统

基于 LangGraph 的 RAG 文档问答系统，支持 PDF/Word 文档上传、多轮问答、SSE 流式输出。

## 技术栈

| 层级 | 技术 |
|------|------|
| LLM | DeepSeek Chat API（兼容 OpenAI SDK） |
| 嵌入模型 | BAAI/bge-small-zh-v1.5（本地，免费） |
| 重排序 | BAAI/bge-reranker-base（本地 Cross-Encoder） |
| 向量存储 | ChromaDB |
| 关键词检索 | BM25（rank-bm25 + jieba 分词） |
| Agent 框架 | LangGraph（ReAct Agent + 条件边路由） |
| 后端 | FastAPI + SSE 流式输出 |
| 前端 | Next.js + React + Markdown 渲染 |
| 部署 | Docker Compose |

## 快速开始

### 1. 配置环境变量

复制 `.env.example` 为 `.env` 并填入你的 DeepSeek API Key（不要直接写在命令行中，避免 Key 泄露到 shell 历史）：

```bash
cp .env.example .env
# 然后编辑 .env，把 DEEPSEEK_API_KEY 改为你本人的真实 Key
```

> 项目根目录已提供 `.env.example` 作为占位模板，`.env` 已被 `.gitignore` 忽略，**严禁提交**。

> **安全警告**: `.env` 文件包含你的 API Key，已加入 `.gitignore`，**严禁提交到 Git**。每次提交前请确认 `git status` 中不包含 `.env`。

### 2. 启动服务

```bash
docker compose up -d
```

首次启动会自动下载模型（BGE Embedding + Reranker），约需 2-5 分钟。

### 3. 访问

- 前端: http://localhost:3000
- 后端 API 文档: http://localhost:8000/docs

## 提交前安全检查

本项目 `.gitignore` 已配置忽略以下敏感文件，提交前务必确认：

```bash
# 检查是否有敏感文件被意外跟踪
git status

# 如果 .env 已被跟踪，从 Git 中移除（保留本地文件）
git rm --cached .env
```

被忽略的敏感文件类型：
- `.env` — 环境变量（含 API Key）
- `chroma_db/` — ChromaDB 持久化数据
- `backend/data/*.pdf` `.docx` `.doc` — 上传的文档
- `model_cache/` — 本地模型缓存

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/upload | 上传 PDF/Word 文档 |
| POST | /api/chat | 非流式问答 |
| POST | /api/chat/stream | SSE 流式问答 |
| GET | /api/index/status | 查询索引状态 |
| DELETE | /api/index | 清空索引 |

## 项目结构

```
smart-doc-qa/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI 入口
│   │   ├── config.py            # 全局配置
│   │   ├── document/            # 文档解析与分块
│   │   ├── retrieval/           # 检索模块（向量+BM25+Rerank）
│   │   └── agent/               # LangGraph Agent
│   ├── data/                    # 上传文档存放（git 忽略）
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── app/                 # Next.js 路由
│   │   ├── components/          # React 组件
│   │   └── lib/                 # API 调用封装
│   ├── package.json
│   └── Dockerfile
├── docs/                        # 样例/示例文档（HR 面试题库等）
├── .env.example                 # 环境变量模板
├── docker-compose.yml
└── README.md
```
