"""Operational CLI: ``python -m app.cli <command>``.

Commands
--------
``init-db``     create tables (idempotent)
``create-user`` create an account non-interactively
``reindex``     dispatch a re-ingestion task for one document
``health``      print the LLM/embedding/vector-store health snapshot
"""

from __future__ import annotations

import argparse
import json
import sys


def _init_db() -> int:
    from app.db.init_db import create_all

    create_all()
    print("Database schema is up to date.")
    return 0


def _create_user(args: argparse.Namespace) -> int:
    from app.db.init_db import create_all
    from app.db.session import session_scope
    from app.schemas.auth import UserCreate
    from app.services.auth_service import AuthService

    create_all()
    with session_scope() as session:
        user = AuthService(session).register(
            UserCreate(
                email=args.email,
                password=args.password,
                full_name=args.full_name,
            ),
            role="admin" if args.admin else "user",
        )
        print(f"Created user {user.email} (role={user.role}, id={user.id})")
    return 0


def _reindex(args: argparse.Namespace) -> int:
    from app.workers.tasks.ingestion import ingest_document

    result = ingest_document.apply_async(args=(args.document_id, None))
    print(f"Dispatched ingestion task {result.id} for document {args.document_id}")
    return 0


def _health() -> int:
    from app.llm.factory import get_llm_provider
    from app.rag.embeddings import get_embedding_provider

    payload = {
        "llm": get_llm_provider().health(),
        "embeddings": get_embedding_provider().health(),
    }
    try:
        from app.db.vector_store import get_vector_store

        payload["vector_store"] = {"vectors": get_vector_store().count(), "status": "ok"}
    except Exception as exc:  # noqa: BLE001 - report, do not crash the CLI
        payload["vector_store"] = {"status": f"error: {exc}"}

    print(json.dumps(payload, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="app.cli", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-db", help="Create database tables.")

    create_user = subparsers.add_parser("create-user", help="Create a user account.")
    create_user.add_argument("--email", required=True)
    create_user.add_argument("--password", required=True)
    create_user.add_argument("--full-name", default=None)
    create_user.add_argument("--admin", action="store_true")

    reindex = subparsers.add_parser("reindex", help="Re-ingest a document.")
    reindex.add_argument("document_id")

    subparsers.add_parser("health", help="Print a dependency health snapshot.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "init-db": lambda _: _init_db(),
        "create-user": _create_user,
        "reindex": _reindex,
        "health": lambda _: _health(),
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
