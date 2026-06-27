# Enterprise Customer Service Agent V2

面向企业售后客服场景的 AI Agent 后端项目，聚焦政策问答、订单查询、退款处理、转人工、知识库检索和工具调用治理。

项目以 FastAPI 为服务入口，结合 OpenAI-compatible 大模型接口、Embedding、Qdrant、Redis、MySQL 和 Function Calling，构建一个可运行、可验证、可继续扩展的智能客服后端。

## 功能概览

- FastAPI 提供聊天接口、SSE 流式接口、Swagger 文档和本地运营验证台。
- OpenAI-compatible Chat Completions API 接入大模型。
- OpenAI-compatible Embeddings API 或 Ollama Embeddings 生成向量。
- Qdrant 作为知识库向量检索引擎。
- Redis 用于语义缓存，减少相似问题重复调用模型。
- MySQL 保存会话、消息、订单和售后工单。
- Function Calling 支持订单查询、退款工单、转人工和公开网页搜索。
- 高风险工具调用会先返回确认提示，避免模型直接执行写操作。
- 用户问题在进入 RAG 和 LLM 前会做轻量标准化。
- 多轮历史过长时会压缩较早上下文，控制模型输入长度。
- 本地验证台可查看回答、知识库证据、工具结果和 trace。
- Docker Compose 提供 MySQL、Redis、Qdrant 本地依赖。
- GitHub Actions 执行基础测试和代码检查。

## 架构图

![企业智能客服 Agent 运行链路](assets/customer-service-runtime.svg)

核心链路：

```text
用户 / 运营验证台
-> FastAPI 路由
-> CustomerServiceAgent
-> 会话与短期上下文
-> 语义缓存
-> RAG 知识库检索
-> LLM 决策
-> 工具调用或直接回答
-> 消息落库并返回响应
```

## 技术栈

- Python 3.11+
- FastAPI
- Pydantic v2
- SQLAlchemy 2.x Async ORM
- MySQL
- Redis
- Qdrant
- OpenAI-compatible Chat Completions API
- OpenAI-compatible Embeddings API
- pytest
- ruff

## 项目结构

```text
enterprise-customer-service-agent-v2/
  Dockerfile
  docker-compose.yml
  pyproject.toml
  requirements.txt
  sample_knowledge/
  scripts/
    check_env.py
    init_db.py
    seed_sample_data.py
    ingest_docs.py
    run_dev.py
  src/customer_service_app/
    api/
    core/
    domain/
    infrastructure/
    prompts/
    services/
    tools/
    workflows/
    web/
  tests/
```

## 配置说明

复制配置模板：

```bash
cp .env.example .env
```

常用配置项：

```dotenv
LLM_API_KEY=
LLM_BASE_URL=
LLM_MODEL=

EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=
EMBEDDING_MODEL=
EMBEDDING_DIMENSION=1024

DATABASE_URL=mysql+aiomysql://customer_service:customer_service@127.0.0.1:3306/customer_service?charset=utf8mb4
REDIS_URL=redis://127.0.0.1:6380/0
QDRANT_URL=http://127.0.0.1:6333
QDRANT_COLLECTION=customer_service_knowledge
```

`.env.example` 只保留空值和示例配置，真实密钥不要提交到仓库。

## 本地启动

创建虚拟环境并安装依赖：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

启动本地依赖：

```bash
docker compose up -d mysql redis qdrant
```

初始化数据：

```bash
python scripts/check_env.py
python scripts/init_db.py
python scripts/seed_sample_data.py
python scripts/ingest_docs.py sample_knowledge --tenant-id default
```

启动服务：

```bash
python scripts/run_dev.py
```

本地访问：

- API Docs: <http://127.0.0.1:8000/docs>
- Ops Console: <http://127.0.0.1:8000/ops>
- Health Check: <http://127.0.0.1:8000/health/ready>

## 示例问题

```text
你好，我想了解一下七天无理由退货政策。
我的订单 202606040001 到哪里了？
我的订单 202606040001 已经签收了，但我想申请退款。
这个问题我需要人工客服处理。
```

## API 示例

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "default",
    "user_id": "u001",
    "conversation_id": null,
    "question": "我的订单 202606040001 到哪里了？",
    "history": [],
    "metadata": {}
  }'
```

## 测试

```bash
pytest -q
ruff check src tests scripts
```

## 后续演进方向

- 引入独立确认接口，承接退款、转人工等高风险动作。
- 引入 LangGraph 多节点编排，增强复杂任务的状态管理。
- 引入 MCP 服务边界，将订单、售后、工单等业务能力从 Agent 主服务中拆分出去。
- 增强长短期记忆、成本治理和运行观测能力。
- 增加更完整的 RAG 评测集和回归测试。

## License

MIT
