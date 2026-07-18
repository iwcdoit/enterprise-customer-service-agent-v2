# Knowledge Base

该目录是客服 Agent 的可检索业务知识库，不包含实时订单、物流或工单状态。实时事实必须通过 Tool/MCP 读取业务系统。

文档按 FAQ、政策、SOP、产品手册、故障排查、权益、公告、规则矩阵、风控、商家规则和 SLA 分类。同一批 chunk 使用相同 ID 写入向量库和 OpenSearch，在查询时通过 RRF 融合排名。

入库命令：

```bash
python scripts/ingest_docs.py knowledge_base --tenant-id <tenant-id>
```

`evaluation_queries.json` 保存离线检索评测集。本说明文件不会被入库。
