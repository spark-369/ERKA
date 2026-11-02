"""Generic repository base class (small, explicit CRUD helper)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from app.db.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    """CRUD primitives shared by every concrete repository.

    Repositories never commit: transaction boundaries belong to the service
    layer (or to :func:`app.db.session.session_scope` in workers).
    """

    model: type[ModelT]

    def __init__(self, session: Session) -> None:
        self.session = session

    # ------------------------------------------------------------ primitives
    def add(self, instance: ModelT) -> ModelT:
        self.session.add(instance)
        self.session.flush()
        return instance

    def add_all(self, instances: Sequence[ModelT]) -> Sequence[ModelT]:
        self.session.add_all(instances)
        self.session.flush()
        return instances

    def get(self, entity_id: str) -> ModelT | None:
        return self.session.get(self.model, entity_id)

    def delete(self, instance: ModelT) -> None:
        self.session.delete(instance)
        self.session.flush()

    def count(self, statement: Select[Any] | None = None) -> int:
        if statement is None:
            return int(self.session.scalar(select(func.count()).select_from(self.model)) or 0)
        subquery = statement.order_by(None).subquery()
        return int(self.session.scalar(select(func.count()).select_from(subquery)) or 0)

    def list(
        self,
        statement: Select[Any] | None = None,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> list[ModelT]:
        stmt = statement if statement is not None else select(self.model)
        result = self.session.execute(stmt.offset(offset).limit(limit)).scalars().all()
        return list(result)

    def first(self, statement: Select[Any]) -> ModelT | None:
        return self.session.execute(statement).scalars().first()

    def refresh(self, instance: ModelT) -> ModelT:
        self.session.refresh(instance)
        return instance

    def commit(self) -> None:
        """Commit the current unit of work."""
        self.session.commit()

    def rollback(self) -> None:
        self.session.rollback()
