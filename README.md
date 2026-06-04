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

```bash
cp .env.example .env
# 编辑 .env，填入你的 DEEPSEEK_API_KEY
```

### 2. 启动服务

```bash
docker compose up -d
```

首次启动会自动下载模型（BGE Embedding + Reranker），约需 2-5 分钟。

### 3. 访问

- 前端: http://localhost:3000
- 后端 API 文档: http://localhost:8000/docs

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
│   ├── data/                    # 上传文档存放
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── app/                 # Next.js 路由
│   │   ├── components/          # React 组件
│   │   └── lib/                 # API 调用封装
│   └── Dockerfile
└── docker-compose.yml
```
