"""Scoped API-key authentication: least privilege for the four things a caller can do.

The scopes exist so that separation of duties in :mod:`bandit_llm_router.activation` means
something: if the key that posts rewards could also activate a policy, "a second principal approved
it" would be theatre. These tests check the four scopes are genuinely independent, that the verifier
fails closed with nothing configured, and that a wrong key reveals nothing through its response.
"""

from __future__ import annotations

from typing import Annotated, Any

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry
from pydantic import ValidationError

from bandit_llm_router.auth import (
    Scope,
    ScopedApiKeyVerifier,
    ScopedPrincipal,
    hash_api_key,
    require_scope,
)
from bandit_llm_router.core.config import Settings
from bandit_llm_router.core.errors import install_error_handlers
from bandit_llm_router.core.metrics import create_http_metrics

pytestmark = pytest.mark.unit

ROUTE_KEY = "route-key-" + "r" * 30
FEEDBACK_KEY = "feedback-key-" + "f" * 30
ADMIN_KEY = "admin-key-" + "a" * 30
APPROVE_KEY = "approve-key-" + "p" * 30


def settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "route_api_key_sha256": [hash_api_key(ROUTE_KEY)],
        "feedback_api_key_sha256": [hash_api_key(FEEDBACK_KEY)],
        "admin_api_key_sha256": [hash_api_key(ADMIN_KEY)],
        "approver_api_key_sha256": [hash_api_key(APPROVE_KEY)],
        "environment": "test",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


# --- the verifier --------------------------------------------------------------------------------


def test_each_key_carries_only_its_own_scope() -> None:
    verifier = ScopedApiKeyVerifier.from_settings(settings())
    for key, scope in (
        (ROUTE_KEY, Scope.ROUTE),
        (FEEDBACK_KEY, Scope.FEEDBACK),
        (ADMIN_KEY, Scope.ADMIN),
        (APPROVE_KEY, Scope.APPROVE),
    ):
        principal = verifier.verify(key)
        assert principal is not None
        assert principal.scopes == {scope}
        for other in Scope:
            assert principal.has(other) is (other is scope)


def test_a_key_listed_under_two_scopes_holds_both() -> None:
    shared = "shared-key-" + "s" * 30
    verifier = ScopedApiKeyVerifier.from_settings(
        settings(
            admin_api_key_sha256=[hash_api_key(shared)],
            approver_api_key_sha256=[hash_api_key(shared)],
        )
    )
    principal = verifier.verify(shared)
    assert principal is not None
    assert principal.scopes == {Scope.ADMIN, Scope.APPROVE}


def test_an_unknown_key_is_simply_unknown() -> None:
    verifier = ScopedApiKeyVerifier.from_settings(settings())
    assert verifier.verify("not-a-key") is None
    assert verifier.verify(None) is None
    assert verifier.verify("") is None
    # An over-long presented value is rejected without hashing megabytes.
    assert verifier.verify("x" * 10_000) is None


def test_the_verifier_fails_closed_with_nothing_configured() -> None:
    verifier = ScopedApiKeyVerifier.from_settings(
        settings(
            route_api_key_sha256=[],
            feedback_api_key_sha256=[],
            admin_api_key_sha256=[],
            approver_api_key_sha256=[],
        )
    )
    assert not verifier.configured
    assert verifier.verify(ROUTE_KEY) is None


def test_the_key_id_is_not_the_key() -> None:
    verifier = ScopedApiKeyVerifier.from_settings(settings())
    principal = verifier.verify(ROUTE_KEY)
    assert principal is not None
    assert len(principal.key_id) == 12
    assert principal.key_id in hash_api_key(ROUTE_KEY)
    assert ROUTE_KEY not in principal.key_id


def test_the_digests_are_exposed_for_the_base_app_but_the_keys_are_not() -> None:
    verifier = ScopedApiKeyVerifier.from_settings(settings())
    digests = verifier.digests
    assert len(digests) == 4
    assert all(len(digest) == 64 for digest in digests)
    assert hash_api_key(ROUTE_KEY) in digests


def test_a_malformed_digest_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="64 hex characters"):
        ScopedApiKeyVerifier({"deadbeef": [Scope.ROUTE]})


def test_the_settings_refuse_a_non_digest() -> None:
    with pytest.raises(ValidationError):
        settings(route_api_key_sha256=["not-a-sha256"])


def test_a_principal_is_hashable_and_frozen() -> None:
    principal = ScopedPrincipal(key_id="abc123", scopes=frozenset({Scope.ROUTE}))
    assert principal.has(Scope.ROUTE)
    with pytest.raises(AttributeError):
        principal.key_id = "other"  # type: ignore[misc]


# --- the FastAPI dependency ----------------------------------------------------------------------

# Module level, because FastAPI resolves a route's annotations against module globals.
RoutePrincipal = Annotated[ScopedPrincipal, Depends(require_scope(Scope.ROUTE))]
AdminPrincipal = Annotated[ScopedPrincipal, Depends(require_scope(Scope.ADMIN))]


def build_app() -> FastAPI:
    """A bare app carrying just the verifier, the metrics and two scoped routes."""
    app = FastAPI()
    app.state.scoped_verifier = ScopedApiKeyVerifier.from_settings(settings())
    app.state.metrics = create_http_metrics(CollectorRegistry())
    install_error_handlers(app)

    @app.get("/route-only")
    async def route_only(principal: RoutePrincipal) -> dict[str, str]:
        return {"key_id": principal.key_id}

    @app.get("/admin-only")
    async def admin_only(principal: AdminPrincipal) -> dict[str, str]:
        return {"key_id": principal.key_id}

    return app


@pytest.fixture
def app() -> FastAPI:
    return build_app()


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def test_a_missing_key_is_a_401_with_a_challenge(client: TestClient) -> None:
    response = client.get("/route-only")
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "ApiKey"
    # The message says nothing about which keys exist.
    assert "API key" in response.json()["error"]["message"]


def test_a_valid_key_without_the_scope_is_a_403(client: TestClient) -> None:
    response = client.get("/admin-only", headers={"X-API-Key": ROUTE_KEY})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"


def test_the_right_scope_is_admitted(client: TestClient) -> None:
    response = client.get("/admin-only", headers={"X-API-Key": ADMIN_KEY})
    assert response.status_code == 200
    assert len(response.json()["key_id"]) == 12


def test_a_bearer_token_is_accepted_too(client: TestClient) -> None:
    response = client.get("/route-only", headers={"Authorization": f"Bearer {ROUTE_KEY}"})
    assert response.status_code == 200


def test_failures_are_counted(app: FastAPI, client: TestClient) -> None:
    counter = app.state.metrics.auth_failures_total
    assert counter._value.get() == 0.0
    client.get("/route-only")
    client.get("/admin-only", headers={"X-API-Key": ROUTE_KEY})
    assert counter._value.get() == 2.0


def test_the_route_and_feedback_scopes_cannot_substitute_for_each_other(
    client: TestClient,
) -> None:
    """Whoever can write rewards must not be able to spend a tenant's budget, and vice versa."""
    assert client.get("/route-only", headers={"X-API-Key": FEEDBACK_KEY}).status_code == 403
    assert client.get("/route-only", headers={"X-API-Key": ROUTE_KEY}).status_code == 200
