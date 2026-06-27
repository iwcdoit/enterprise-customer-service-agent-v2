from __future__ import annotations


class AppError(Exception):
    """Base class for expected application errors converted to JSON responses."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, code: str | None = None, status_code: int | None = None):
        """Create an application error with an optional code and HTTP status."""
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        if status_code:
            self.status_code = status_code


class ConfigurationError(AppError):
    """配置错误：比如 API Key、数据库 URL 没有填写。"""

    status_code = 500
    code = "configuration_error"


class ExternalServiceError(AppError):
    """外部服务错误：比如 LLM、Qdrant、Redis 请求失败。"""

    status_code = 502
    code = "external_service_error"


class NotFoundError(AppError):
    """资源不存在。"""

    status_code = 404
    code = "not_found"


class PermissionDeniedError(AppError):
    """权限不足或认证失败。"""

    status_code = 403
    code = "permission_denied"
