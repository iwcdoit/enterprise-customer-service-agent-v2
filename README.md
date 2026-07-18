# Enterprise Customer Service Agent V2

一个面向消费者和企业客户服务场景的 AI Agent 后端项目。当前重点是会话、有界任务规划、混合知识检索、工具调用、LangGraph 人工确认与恢复、MCP 业务边界、成本策略和结果落库。

相比 V1 版本，V2 围绕 LangGraph 编排、HIL（checkpoint / interrupt / resume）、MCP 业务服务边界、短期与长期记忆、模型成本治理，以及观测与评估能力持续演进。

## 架构

当前运行链路：

```mermaid
flowchart TD
    API["FastAPI / Chat API"] --> AGENT["CustomerServiceAgent"]
    AGENT -->|"人工已接管"| HUMAN["人工队列与消息落库"]
    AGENT -->|"Bot 模式"| START((START))
    START --> PREPARE["prepare\n校验会话 + 零 Token Planner Gate"]
    PREPARE -->|"简单请求"| ANSWER["answer\n标准回答链路"]
    PREPARE -->|"复杂多意图"| PLAN["plan\n有界结构化计划"]
    PLAN --> EXECUTE["execute_plan\nReAct 执行与步数/超时限制"]
    EXECUTE -->|"无高风险动作"| ANSWER
    EXECUTE -->|"退款、补偿、转人工"| INTERRUPT["await_confirmation\nCheckpoint + interrupt"]
    INTERRUPT -. "approve / reject" .-> RESUME["apply_confirmation\nCommand(resume=...)"]
    RESUME -->|"仍有待确认动作"| INTERRUPT
    RESUME --> ANSWER

    subgraph TURN["标准回答链路"]
      ANSWER --> COST["成本策略 + 短期历史"]
      COST --> CACHE["Redis 语义缓存"]
      CACHE -->|"命中"| PERSIST
      CACHE -->|"未命中"| RETRIEVE["向量召回 || BM25 召回"]
      RETRIEVE --> FUSION["RRF 融合 + 可选 Reranker"]
      FUSION --> LLM["LLM + tools"]
      LLM -->|"tool_calls"| GATEWAY["ToolRegistry -> BusinessGateway"]
      GATEWAY -->|"本地或独立服务"| MCP["after-sales MCP"]
      MCP --> LLM
      LLM --> PERSIST["消息 / Trace / Token 用量落库"]
    end
    PERSIST --> END((END))
```

核心原则是让大模型负责理解和决策，后端工具与 MCP 服务负责查询真实事实和执行业务动作。高风险操作不由模型直接落库，而是经过租户/用户校验、Graph HIL、幂等键和审计关联后执行。

## 当前能力

- LangGraph Checkpoint、`interrupt` 与恢复，确认单与 Graph thread 绑定。
- 复杂请求触发有界 Planner/ReAct，简单问题跳过规划模型。
- Qdrant/Milvus 向量召回与 OpenSearch BM25 并行召回，通过 RRF 融合，并支持可选 HTTP Reranker。
- Markdown 标题、段落、句子边界和重叠窗口切块，向量索引与 BM25 索引共用 chunk ID。
- Tool Registry、Business Gateway 和独立 after-sales MCP 服务，读写工具分级治理。
- 人工队列、模拟坐席接管、处理结论和 Bot 恢复闭环。
- 租户级模型、历史窗口、RAG 数量、Rerank 和缓存策略调整。

## 成本治理

V2 的成本治理目标不是简单拒绝服务，而是在租户日用量接近或超过预算时，动态调整运行策略，尽量保持客服链路可用：

- 降级模型时同步降低任务复杂度，减少复杂规划、多工具链和长推理，优先使用确定性规则完成可控判断。
- 低成本模型使用更明确的结构化提示词，降低模型自由发挥空间，提高意图识别和工具参数生成的稳定性。
- 工具调用继续使用严格参数校验，缺少订单号、原因、确认 token 等关键字段时先追问或中断，不直接执行业务动作。
- 对包含“这个、那个、刚才、上面”等模糊指代的问题，优先做问题改写或补充短期记忆，避免低成本模型误解上下文。
- 高风险动作不随模型降级而降低安全标准，退款、补偿、换货、转人工等操作仍然需要人工确认、幂等控制和审计记录。
- 成本策略同时调整模型、RAG 召回数量、历史窗口、rerank 和缓存优先级，而不是只替换一个更便宜的模型名称。

## 技术栈

- Python 3.11+
- FastAPI
- Pydantic v2
- SQLAlchemy 2.x Async ORM
- MySQL
- Redis
- Qdrant / Milvus
- OpenSearch BM25
- HTTP Reranker adapter
- LangGraph
- MCP Python SDK
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
  assets/
  knowledge_base/
  scripts/
    check_env.py
    init_db.py
    seed_sample_data.py
    ingest_docs.py
    run_dev.py
    run_after_sales_mcp_server.py
  mcp_services/
    after_sales_server/
  src/customer_service_app/
    api/
    core/
    domain/
    infrastructure/
      cache/
      db/
      embeddings/
      knowledge_ingestion/
      lexical_search/
      llm/
      mcp/
      rerank/
      search/
      vector_store/
    prompts/
    services/
    tools/
    workflows/
    web/
  tests/
```

## 配置

复制环境变量模板：

```bash
cp .env.example .env
```

主要配置：

```dotenv
RUNTIME_ENV=production
ALLOWED_ORIGINS=https://console.example.com

LLM_PROVIDER=openai_compatible
LLM_API_KEY=<your-llm-api-key>
LLM_BASE_URL=https://llm-provider.example.com/v1
LLM_MODEL=<chat-model-name>

EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_API_KEY=<your-embedding-api-key>
EMBEDDING_BASE_URL=https://embedding-provider.example.com/v1
EMBEDDING_MODEL=<embedding-model-name>
EMBEDDING_DIMENSION=1024

DATABASE_URL=mysql+aiomysql://<user>:<password>@<mysql-host>:3306/<database>?charset=utf8mb4
REDIS_URL=redis://:<password>@<redis-host>:6379/0

VECTOR_STORE_PROVIDER=qdrant
QDRANT_URL=https://qdrant.example.com
QDRANT_API_KEY=<your-qdrant-api-key>
QDRANT_COLLECTION=customer_service_knowledge

MILVUS_URI=https://milvus.example.com
MILVUS_TOKEN=<your-milvus-token>
MILVUS_COLLECTION=customer_service_knowledge

RETRIEVAL_MODE=hybrid
OPENSEARCH_ENABLED=true
OPENSEARCH_URL=https://<opensearch-host>:9200
OPENSEARCH_INDEX=customer_service_knowledge

RERANK_ENABLED=false
RERANK_BASE_URL=https://<rerank-provider>/v1/rerank
RERANK_MODEL=<rerank-model-name>

MCP_AFTER_SALES_ENABLED=true
MCP_AFTER_SALES_URL=https://mcp-after-sales.example.com/mcp
MCP_APPROVAL_SIGNING_SECRET=<random-signing-secret>

```

## 部署说明

主服务可以作为 FastAPI 应用部署，并依赖以下独立基础设施和业务服务：

- MySQL：保存会话、消息、订单、售后工单、确认记录、审计日志和长期记忆。
- Redis：承接语义缓存、运行锁、限流策略和轻量计数。
- Qdrant / Milvus：承接客户服务知识的向量检索。
- OpenSearch：承接规则编号、产品名和专有词等 BM25 关键词检索。
- Reranker（可选）：对融合后的少量候选块做二次排序。
- after-sales MCP service：提供订单、物流、退款、补偿、换货、转人工等业务能力。
- LangSmith / Prometheus / Grafana：提供链路追踪、指标、看板和运行分析。

典型启动顺序：

```bash
python scripts/check_env.py
python scripts/init_db.py
python scripts/seed_sample_data.py
python scripts/ingest_docs.py knowledge_base --tenant-id <tenant-id>
python scripts/run_after_sales_mcp_server.py
uvicorn customer_service_app.main:app --host 0.0.0.0 --port <port>
```

完成入库后，可用真实检索链路计算 Hit@K 和 MRR：

```bash
python scripts/evaluate_retrieval.py --tenant-id <tenant-id> --top-k 5
python scripts/evaluate_retrieval.py --tenant-id <tenant-id> --top-k 5 --rerank
```

容器化部署时，通过目标平台注入环境变量，并使用项目中的 Dockerfile 或 Compose 模板启动服务。

## MCP 服务

after-sales MCP 服务将售后能力暴露为独立治理的业务工具：

- `query_order_status`
- `query_logistics_status`
- `query_price_protection`
- `query_customer_profile`
- `create_refund_case`
- `create_compensation_case`
- `create_exchange_case`
- `transfer_to_human`

只读工具在完成租户和用户校验后可以直接执行。写入工具需要签名确认 token 和幂等键，避免同一个已确认动作被重复执行。

## API 示例

```bash
curl -X POST "https://api.example.com/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "tenant_001",
    "user_id": "user_001",
    "conversation_id": null,
    "question": "我的订单 202606040001 已经签收了，但是商品有破损，我想申请退款或者补偿。",
    "history": [],
    "metadata": {}
  }'
```

## 测试

```bash
pytest -q
ruff check src tests scripts mcp_services/after_sales_server/src
```

## License

MIT
