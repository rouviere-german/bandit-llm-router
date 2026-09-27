"""Scoped API-key authentication: routing, feedback, administration and approval are separate.

The baseline verifier in ``core.security`` answers "is this a valid key?". That is not enough here,
because the four things callers can do have very different blast radii:

* ``route`` -- spend a tenant's budget on model calls.
* ``feedback`` -- write rewards, which is to say, steer what the policy learns.
* ``admin`` -- propose and activate policies.
* ``approve`` -- approve someone else's activation.

A key carries only the scopes it was configured with, so the service that posts rewards cannot
activate a policy and the reviewer's key cannot route traffic. Separation of duties in
:mod:`bandit_llm_router.activation` depends on this: the approver must be a *different* principal
from the proposer, and that is only meaningful if the scopes are distinct.

Keys are held as SHA-256 digests, never in plaintext, and a presented key is compared against every
configured digest with :func:`hmac.compare_digest` and no early exit, so response timing reveals
neither which key matched nor how far a guess got. With no keys configured the verifier fails
closed: every scoped route rejects.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from fastapi import Request

from bandit_llm_router.core.errors import AppError
from bandit_llm_router.core.security import extract_api_key, hash_api_key

if TYPE_CHECKING:
    from collections.abc import Callable

    from bandit_llm_router.core.config import Settings

__all__ = [
    "Scope",
    "ScopedApiKeyVerifier",
    "ScopedPrincipal",
    "hash_api_key",
    "require_scope",
]

_MAX_KEY_LENGTH = 512


class Scope(StrEnum):
    """What a key is allowed to do."""

    ROUTE = "route"
    FEEDBACK = "feedback"
    ADMIN = "admin"
    APPROVE = "approve"


@dataclass(frozen=True, slots=True)
class ScopedPrincipal:
    """An authenticated caller: a non-secret key id plus the scopes it holds."""

    key_id: str
    scopes: frozenset[Scope]

    def has(self, scope: Scope) -> bool:
        return scope in self.scopes


class ScopedApiKeyVerifier:
    """Maps key digests to scopes and verifies a presented key in constant time."""

    def __init__(self, digest_scopes: Mapping[str, Iterable[Scope]]) -> None:
        entries: list[tuple[bytes, frozenset[Scope]]] = []
        for digest, scopes in digest_scopes.items():
            if len(digest) != 64:
                raise ValueError("API key digests must be 64 hex characters (SHA-256)")
            entries.append((bytes.fromhex(digest), frozenset(scopes)))
        self._entries = tuple(entries)

    @classmethod
    def from_settings(cls, settings: Settings) -> ScopedApiKeyVerifier:
        """Build the verifier from the four per-scope digest lists in settings."""
        mapping: dict[str, set[Scope]] = {}
        for scope, digests in (
            (Scope.ROUTE, settings.route_api_key_sha256),
            (Scope.FEEDBACK, settings.feedback_api_key_sha256),
            (Scope.ADMIN, settings.admin_api_key_sha256),
            (Scope.APPROVE, settings.approver_api_key_sha256),
        ):
            for digest in digests:
                mapping.setdefault(digest, set()).add(scope)
        return cls({digest: frozenset(scopes) for digest, scopes in mapping.items()})

    @property
    def configured(self) -> bool:
        return bool(self._entries)

    @property
    def digests(self) -> tuple[str, ...]:
        """The configured digests, for the core app's rate limiter and metrics auth."""
        return tuple(digest.hex() for digest, _ in self._entries)

    def verify(self, presented: str | None) -> ScopedPrincipal | None:
        """Return the principal for a valid key, else ``None``. No early exit on a match."""
        if not presented or len(presented) > _MAX_KEY_LENGTH:
            return None
        digest = hashlib.sha256(presented.encode("utf-8")).digest()
        matched: tuple[bytes, frozenset[Scope]] | None = None
        for candidate, scopes in self._entries:
            if hmac.compare_digest(digest, candidate):
                matched = (candidate, scopes)
        if matched is None:
            return None
        return ScopedPrincipal(key_id=matched[0].hex()[:12], scopes=matched[1])


def require_scope(scope: Scope) -> Callable[[Request], ScopedPrincipal]:
    """FastAPI dependency factory: authenticate and require ``scope``.

    A missing or unknown key is a 401; a valid key without the scope is a 403. Both responses are
    fixed strings that say nothing about which keys exist.
    """

    def dependency(request: Request) -> ScopedPrincipal:
        verifier: ScopedApiKeyVerifier = request.app.state.scoped_verifier
        principal = verifier.verify(extract_api_key(request.headers))
        if principal is None:
            request.app.state.metrics.auth_failures_total.inc()
            raise AppError(
                401,
                "unauthorized",
                "A valid API key is required.",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        if not principal.has(scope):
            request.app.state.metrics.auth_failures_total.inc()
            raise AppError(403, "forbidden", "This key is not allowed to perform that action.")
        return principal

    return dependency
