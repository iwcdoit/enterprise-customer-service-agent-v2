from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from customer_service_app.core.config import Settings
from customer_service_app.core.exceptions import ConfigurationError, PermissionDeniedError


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """校验明文密码是否匹配哈希密码。"""
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    """把明文密码转成 bcrypt 哈希，生产中不要存明文密码。"""
    return pwd_context.hash(password)


def create_access_token(subject: str, settings: Settings, extra: dict[str, Any] | None = None) -> str:
    """生成 JWT 登录令牌。

    `subject` 通常放用户 id；`extra` 可以附加租户、角色等业务字段。
    """
    secret = settings.require("JWT_SECRET_KEY", settings.jwt_secret_key)
    expire_at = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    payload: dict[str, Any] = {"sub": subject, "exp": expire_at}
    if extra:
        payload.update(extra)
    return jwt.encode(payload, secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str, settings: Settings) -> dict[str, Any]:
    """解析并校验 JWT，失败时抛权限异常。"""
    if not settings.jwt_secret_key:
        raise ConfigurationError("JWT_SECRET_KEY is required before auth endpoints can be used")
    try:
        return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise PermissionDeniedError("Invalid or expired token") from exc
