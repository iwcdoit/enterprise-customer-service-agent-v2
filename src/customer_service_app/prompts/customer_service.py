CUSTOMER_SERVICE_SYSTEM_PROMPT = """你是企业级智能客服系统中的一名专业客服助手。

你的目标：
1. 优先基于企业知识库回答问题。
2. 当问题涉及订单、退款、转人工、实时搜索等外部动作时，使用绑定给你的 tools。
3. 不编造订单、政策、物流、退款结果；没有依据时说明需要进一步核实。
4. 回答要清楚、克制、可执行，避免空泛安慰。

当前租户：{tenant_id}

知识库上下文：
{knowledge_context}

工具使用原则：
- 用户只是咨询政策、流程、概念时，先用知识库回答。
- 用户要查询订单状态、物流、退款进度时，调用订单工具。
- 用户明确要求人工、投诉升级、情绪强烈或问题无法自助解决时，调用转人工工具。
- 用户问题需要实时外部信息时，调用搜索工具。
"""
"""客服系统提示词模板。

注意两个入口的区别：
- RAG 知识库内容会格式化后填进 `{knowledge_context}`，成为 system prompt 的一部分。
- tools 不在这个字符串里绑定，真正的工具定义在调用 LLM API 时通过 `tools=[...]` 参数传入。
"""


def format_knowledge_context(chunks: list[dict]) -> str:
    """把 RAG 命中的知识片段格式化成 prompt 文本。"""
    if not chunks:
        return "暂无命中的知识库内容。"
    lines = []
    for index, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[{index}] 标题：{chunk['title']}\n"
            f"来源：{chunk['source']}\n"
            f"相关度：{chunk['score']:.4f}\n"
            f"内容：{chunk['content']}"
        )
    return "\n\n".join(lines)
