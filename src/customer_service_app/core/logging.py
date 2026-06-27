from __future__ import annotations

import json
import logging
import logging.config
from datetime import datetime, timezone
from typing import Any

from customer_service_app.core.config import Settings


class JsonFormatter(logging.Formatter):
    """把普通日志格式化成 JSON，便于后续接入 ELK / Loki / 云日志。"""

    def format(self, record: logging.LogRecord) -> str:
        """把一条日志记录转换成 JSON 字符串。

        生产注意：默认不打印完整 question/answer，避免泄露用户隐私。
        如果需要排查单次链路，应优先看接口返回的 trace 或增加脱敏后的业务日志。
        """
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra_fields"):
            payload.update(record.extra_fields)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(settings: Settings) -> None:
    """初始化应用日志配置。"""
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"json": {"()": JsonFormatter}},
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": "json",
                }
            },
            "root": {
                "level": settings.log_level,
                "handlers": ["default"],
            },
        }
    )


def get_logger(name: str) -> logging.Logger:
    """获取 logger，业务代码通过它打印结构化日志。"""
    return logging.getLogger(name)
