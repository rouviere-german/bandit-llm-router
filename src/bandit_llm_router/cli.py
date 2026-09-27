"""Command line entry point: ``bandit-llm-router`` or ``python -m bandit_llm_router``.

Subcommands:

* ``serve`` -- run the routing service (uvicorn).
* ``demo`` -- the offline simulated day (no API key, no network).
* ``generate-api-key`` -- print a new key and its SHA-256 digest for one of the four scopes.
* ``audit verify`` -- verify the hash-chained policy-activation audit log.

Imports are lazy on purpose: ``generate-api-key`` should not pay for numpy, scikit-learn or DuckDB.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__all__ = ["main"]

_SCOPE_ENV = {
    "route": "APP_ROUTE_API_KEY_SHA256",
    "feedback": "APP_FEEDBACK_API_KEY_SHA256",
    "admin": "APP_ADMIN_API_KEY_SHA256",
    "approve": "APP_APPROVER_API_KEY_SHA256",
}


def _serve(host: str | None, port: int | None) -> int:
    import uvicorn

    from bandit_llm_router.core.config import get_settings
    from bandit_llm_router.core.logging import configure_logging
    from bandit_llm_router.core.tracing import install_global_tracer_provider
    from bandit_llm_router.service import create_router_app

    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)
    app = create_router_app(settings)
    install_global_tracer_provider(app.state.tracer_provider)
    uvicorn.run(
        app,
        host=host or settings.host,
        port=port or settings.port,
        log_config=None,
        access_log=False,
        server_header=False,
    )
    return 0


def _demo(workdir: Path) -> int:
    from bandit_llm_router.core.logging import configure_logging
    from bandit_llm_router.demo import run_demo

    # Service logs go to stderr so stdout stays a clean report plus one JSON summary.
    configure_logging("WARNING", json_output=True, stream=sys.stderr)
    summary = run_demo(workdir)
    print(json.dumps({"demo": "bandit-llm-router", **summary}, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bandit-llm-router")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the routing service")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    keys = sub.add_parser("generate-api-key", help="print a new API key and its SHA-256 digest")
    keys.add_argument("--scope", choices=sorted(_SCOPE_ENV), default="route")

    audit = sub.add_parser("audit", help="policy-activation audit log tools")
    audit_sub = audit.add_subparsers(dest="audit_command", required=True)
    verify = audit_sub.add_parser("verify", help="verify a hash-chained audit log")
    verify.add_argument("path", type=Path)
    verify.add_argument("--expect-head", default=None)

    demo = sub.add_parser("demo", help="run the offline simulated day")
    demo.add_argument("--workdir", type=Path, default=Path("var/demo"))

    args = parser.parse_args(argv)
    if args.command == "serve":
        return _serve(args.host, args.port)
    if args.command == "generate-api-key":
        from bandit_llm_router.core.security import generate_api_key

        raw, digest = generate_api_key()
        print(json.dumps({"scope": args.scope, "api_key": raw, "sha256": digest}, indent=2))
        print(
            f"Store only the digest, in {_SCOPE_ENV[args.scope]} (comma-separated).",
            file=sys.stderr,
        )
        return 0
    if args.command == "audit":
        from bandit_llm_router.agentic.audit import main as audit_main

        forwarded = ["verify", str(args.path)]
        if args.expect_head:
            forwarded += ["--expect-head", args.expect_head]
        return audit_main(forwarded)
    return _demo(args.workdir)


if __name__ == "__main__":
    raise SystemExit(main())
