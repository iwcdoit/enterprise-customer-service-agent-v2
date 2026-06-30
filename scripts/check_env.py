from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# 直接运行 scripts/*.py 时，Python 默认不一定知道 `src` 是源码根目录。
# 把 `src` 临时加入 import 路径后，下面才能导入 customer_service_app。

from customer_service_app.core.config import get_settings  # noqa: E402


def _has_value(value: str | None) -> bool:
    """判断配置项是否填写。"""
    return bool(value and value.strip())


def _print_group(title: str, items: list[tuple[str, bool, str]]) -> list[str]:
    """打印一组配置检查结果，并返回缺失项名称。"""
    missing: list[str] = []
    print(f"\n[{title}]")
    for name, ok, note in items:
        mark = "OK" if ok else "MISSING"
        print(f"- {mark:7} {name:28} {note}")
        if not ok:
            missing.append(name)
    return missing


def main() -> int:
    """检查当前 .env/环境变量是否足够启动项目。

    这个脚本只显示配置是否存在，不打印密钥原文。
    """
    parser = argparse.ArgumentParser(description="Check runtime configuration without printing secrets.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when required config is missing.",
    )
    args = parser.parse_args()

    settings = get_settings()
    problems: list[str] = []

    print("Enterprise Customer Service configuration check")
    print(f"runtime_env={settings.runtime_env}")
    print("Secret values are intentionally hidden.")

    problems.extend(
        _print_group(
            "Core API",
            [
                ("APP_NAME", _has_value(settings.app_name), "FastAPI title"),
                ("API_PREFIX", _has_value(settings.api_prefix), "HTTP API prefix"),
            ],
        )
    )
    problems.extend(
        _print_group(
            "LLM",
            [
                ("LLM_API_KEY", _has_value(settings.llm_api_key), "required for real chat"),
                ("LLM_BASE_URL", _has_value(settings.llm_base_url), "OpenAI-compatible endpoint"),
                ("LLM_MODEL", _has_value(settings.llm_model), "chat model name"),
            ],
        )
    )
    problems.extend(
        _print_group(
            "Database",
            [
                ("DATABASE_URL", _has_value(settings.database_url), "required for conversations/tools"),
            ],
        )
    )

    if settings.rag_enabled:
        vector_provider = settings.vector_store_provider.strip().lower()
        vector_checks = (
            [("MILVUS_URI", _has_value(settings.milvus_uri), "Milvus endpoint")]
            if vector_provider == "milvus"
            else [("QDRANT_URL", _has_value(settings.qdrant_url), "Qdrant endpoint")]
        )
        embedding_api_key_ok = True
        if settings.embedding_provider.lower() in {
            "openai_compatible",
            "deepseek",
            "openai",
            "dashscope",
            "bailian",
            "aliyun",
        }:
            embedding_api_key_ok = _has_value(settings.embedding_api_key) or _has_value(
                settings.llm_api_key
            )
        problems.extend(
            _print_group(
                "RAG",
                [
                    ("EMBEDDING_BASE_URL", _has_value(settings.embedding_base_url), "embedding service"),
                    ("EMBEDDING_MODEL", _has_value(settings.embedding_model), "embedding model name"),
                    (
                        "EMBEDDING_API_KEY or LLM_API_KEY",
                        embedding_api_key_ok,
                        "required for cloud embedding",
                    ),
                    ("VECTOR_STORE_PROVIDER", vector_provider in {"qdrant", "milvus"}, "qdrant or milvus"),
                    *vector_checks,
                ],
            )
        )
    else:
        print("\n[RAG]\n- SKIP    RAG_ENABLED=false")

    if settings.semantic_cache_enabled:
        problems.extend(
            _print_group(
                "Semantic Cache",
                [("REDIS_URL", _has_value(settings.redis_url), "semantic cache storage")],
            )
        )
    else:
        print("\n[Semantic Cache]\n- SKIP    SEMANTIC_CACHE_ENABLED=false")

    print("\nResult")
    if problems:
        print("Missing configuration:")
        for item in problems:
            print(f"- {item}")
    else:
        print("All enabled runtime dependencies are configured.")

    return 1 if args.strict and problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
