from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from jose import JWTError, jwt
from passlib.context import CryptContext

from customer_service_app.core.config import Settings
from customer_service_app.core.exceptions import ConfigurationError, PermissionDeniedError


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
PrincipalRole = Literal["customer", "agent", "operator", "admin"]
KNOWN_ROLES: frozenset[str] = frozenset({"customer", "agent", "operator", "admin"})


@dataclass(frozen=True, slots=True)
class CurrentPrincipal:
    """通过可信身份凭证解析出的当前调用者。"""

    subject: str
    tenant_id: str
    roles: frozenset[str]
    authenticated: bool = True

    def has_any_role(self, *roles: str) -> bool:
        """判断调用者是否拥有任意一个目标角色。"""

        return bool(self.roles.intersection(roles))

    @property
    def is_platform_admin(self) -> bool:
        """平台管理员可以跨租户排障，普通租户角色不允许。"""

        return "admin" in self.roles


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
    if settings.jwt_issuer:
        payload["iss"] = settings.jwt_issuer
    if settings.jwt_audience:
        payload["aud"] = settings.jwt_audience
    if extra:
        payload.update(extra)
    return jwt.encode(payload, secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str, settings: Settings) -> dict[str, Any]:
    """解析并校验 JWT，失败时抛权限异常。"""
    if not settings.jwt_secret_key:
        raise ConfigurationError("JWT_SECRET_KEY is required before auth endpoints can be used")
    try:
        options = {
            "verify_iss": bool(settings.jwt_issuer),
            "verify_aud": bool(settings.jwt_audience),
        }
        return jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            issuer=settings.jwt_issuer or None,
            audience=settings.jwt_audience or None,
            options=options,
        )
    except JWTError as exc:
        raise PermissionDeniedError("Invalid or expired token") from exc


def principal_from_claims(claims: dict[str, Any]) -> CurrentPrincipal:
    """把已验签 JWT claims 转成受控的业务身份。"""

    subject = str(claims.get("sub") or "").strip()
    tenant_id = str(claims.get("tenant_id") or claims.get("tid") or "").strip()
    if not subject or not tenant_id:
        raise PermissionDeniedError("Token must contain sub and tenant_id")

    raw_roles = claims.get("roles", claims.get("role", []))
    if isinstance(raw_roles, str):
        roles = {item.strip() for item in raw_roles.split(",") if item.strip()}
    elif isinstance(raw_roles, list):
        roles = {str(item).strip() for item in raw_roles if str(item).strip()}
    else:
        roles = set()
    roles.intersection_update(KNOWN_ROLES)
    if not roles:
        roles = {"customer"}

    return CurrentPrincipal(
        subject=subject,
        tenant_id=tenant_id,
        roles=frozenset(roles),
    )


def authorize_identity(
    principal: CurrentPrincipal,
    *,
    tenant_id: str,
    user_id: str | None = None,
) -> tuple[str, str | None]:
    """校验资源归属，并返回后续业务层应使用的租户和用户身份。"""

    if not principal.authenticated:
        return tenant_id, user_id

    if principal.is_platform_admin:
        return tenant_id, user_id
    if principal.tenant_id != tenant_id:
        raise PermissionDeniedError("Cross-tenant access is forbidden")
    if (
        user_id is not None
        and principal.has_any_role("customer")
        and not principal.has_any_role("agent", "operator")
        and principal.subject != user_id
    ):
        raise PermissionDeniedError("Users may only access their own resources")
    return principal.tenant_id, user_id


def authorize_actor(principal: CurrentPrincipal, *, actor_id: str) -> str:
    """限制普通坐席只能以自己的身份执行人工客服动作。"""

    if not principal.authenticated or principal.has_any_role("operator", "admin"):
        return actor_id
    if principal.subject != actor_id:
        raise PermissionDeniedError("Agent may not act as another operator")
    return principal.subject
