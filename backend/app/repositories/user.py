"""User repository: account lookups used by auth and ownership checks."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select

from app.models.user import User
from app.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    """Queries scoped to the ``users`` table."""

    model = User

    def get_by_email(self, email: str) -> User | None:
        """Case-insensitive lookup (emails are stored lowercased on create)."""
        statement = select(User).where(func.lower(User.email) == email.strip().lower())
        return self.first(statement)

    def email_exists(self, email: str) -> bool:
        return self.get_by_email(email) is not None

    def create(
        self,
        *,
        email: str,
        hashed_password: str,
        full_name: str | None = None,
        role: str = "user",
        is_active: bool = True,
    ) -> User:
        user = User(
            email=email.strip().lower(),
            hashed_password=hashed_password,
            full_name=full_name,
            role=role,
            is_active=is_active,
        )
        return self.add(user)

    def list_users(
        self, *, offset: int = 0, limit: int = 50, only_active: bool = False
    ) -> list[User]:
        statement = select(User)
        if only_active:
            statement = statement.where(User.is_active.is_(True))
        statement = statement.order_by(User.created_at.desc())
        return self.list(statement, offset=offset, limit=limit)

    def count_users(self, *, only_active: bool = False) -> int:
        statement = select(User)
        if only_active:
            statement = statement.where(User.is_active.is_(True))
        return self.count(statement)

    def touch_login(self, user: User) -> User:
        """Record a successful login timestamp."""
        user.last_login_at = datetime.now(timezone.utc)
        self.session.flush()
        return user

    def set_password(self, user: User, hashed_password: str) -> User:
        user.hashed_password = hashed_password
        self.session.flush()
        return user

    def deactivate(self, user: User) -> User:
        user.is_active = False
        self.session.flush()
        return user
