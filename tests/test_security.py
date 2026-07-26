from __future__ import annotations

import pytest

from customer_service_app.api.dependencies import get_current_principal
from customer_service_app.core.config import Settings
from customer_service_app.core.exceptions import AuthenticationError, PermissionDeniedError
from customer_service_app.core.security import (
    CurrentPrincipal,
    authorize_actor,
    authorize_identity,
    create_access_token,
    decode_access_token,
    principal_from_claims,
)


def test_access_token_builds_tenant_principal() -> None:
    settings = Settings(
        security_enabled=True,
        jwt_secret_key="test-secret-with-enough-entropy",
        jwt_issuer="customer-service",
        jwt_audience="customer-api",
    )
    token = create_access_token(
        "user-001",
        settings,
        extra={"tenant_id": "tenant-a", "roles": ["customer"]},
    )

    claims = decode_access_token(token, settings)
    principal = principal_from_claims(claims)

    assert principal.subject == "user-001"
    assert principal.tenant_id == "tenant-a"
    assert principal.roles == frozenset({"customer"})


def test_customer_cannot_cross_tenant_or_read_another_user() -> None:
    principal = CurrentPrincipal(
        subject="user-001",
        tenant_id="tenant-a",
        roles=frozenset({"customer"}),
    )

    with pytest.raises(PermissionDeniedError):
        authorize_identity(
            principal,
            tenant_id="tenant-b",
            user_id="user-001",
        )
    with pytest.raises(PermissionDeniedError):
        authorize_identity(
            principal,
            tenant_id="tenant-a",
            user_id="user-002",
        )


def test_operator_can_access_users_only_inside_own_tenant() -> None:
    principal = CurrentPrincipal(
        subject="operator-001",
        tenant_id="tenant-a",
        roles=frozenset({"operator"}),
    )

    assert authorize_identity(
        principal,
        tenant_id="tenant-a",
        user_id="user-002",
    ) == ("tenant-a", "user-002")
    with pytest.raises(PermissionDeniedError):
        authorize_identity(
            principal,
            tenant_id="tenant-b",
            user_id="user-002",
        )


def test_agent_cannot_impersonate_another_agent() -> None:
    principal = CurrentPrincipal(
        subject="agent-001",
        tenant_id="tenant-a",
        roles=frozenset({"agent"}),
    )

    assert authorize_actor(principal, actor_id="agent-001") == "agent-001"
    with pytest.raises(PermissionDeniedError):
        authorize_actor(principal, actor_id="agent-002")


def test_security_enabled_requires_bearer_token() -> None:
    settings = Settings(security_enabled=True, jwt_secret_key="test-secret")

    with pytest.raises(AuthenticationError):
        get_current_principal(credentials=None, settings=settings)


def test_security_disabled_keeps_local_development_compatible() -> None:
    principal = get_current_principal(
        credentials=None,
        settings=Settings(security_enabled=False),
    )

    assert principal.authenticated is False
    assert authorize_identity(
        principal,
        tenant_id="local-tenant",
        user_id="local-user",
    ) == ("local-tenant", "local-user")
