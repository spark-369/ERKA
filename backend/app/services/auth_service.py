"""Authentication business logic: registration, login, token rotation."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    AuthenticationError,
    ConflictError,
    NotFoundError,
    PermissionDeniedError,
)
from app.core.logging import get_logger
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.models.user import User
from app.repositories.user import UserRepository
from app.schemas.auth import LoginRequest, TokenPair, UserCreate, UserRead, UserUpdate

logger = get_logger(__name__)


class AuthService:
    """Owns every credential-related operation (request-scoped)."""

    def __init__(self, session: Session, config: Settings | None = None) -> None:
        self._session = session
        self._config = config or get_settings()
        self._users = UserRepository(session)

    # ------------------------------------------------------------ registration
    def register(self, payload: UserCreate, *, role: str = "user") -> User:
        """Create a new account (email uniqueness enforced case-insensitively)."""
        if self._users.email_exists(payload.email):
            raise ConflictError(
                "An account with this email already exists.", code="email_taken"
            )

        user = self._users.create(
            email=payload.email,
            hashed_password=hash_password(payload.password),
            full_name=payload.full_name,
            role=role,
        )
        self._session.commit()
        self._session.refresh(user)
        logger.info("Registered user %s (role=%s)", user.email, user.role)
        return user

    # ------------------------------------------------------------------- login
    def authenticate(self, payload: LoginRequest) -> User:
        """Validate credentials and return the user, or raise 401."""
        user = self._users.get_by_email(payload.email)
        if user is None or not verify_password(payload.password, user.hashed_password):
            # Identical error for unknown email and wrong password (no enumeration).
            logger.warning("Failed login attempt for %s", payload.email)
            raise AuthenticationError(
                "Incorrect email or password.", code="invalid_credentials"
            )
        if not user.is_active:
            raise PermissionDeniedError(
                "This account has been deactivated.", code="account_inactive"
            )

        self._users.touch_login(user)
        self._session.commit()
        self._session.refresh(user)
        return user

    def login(self, payload: LoginRequest) -> TokenPair:
        """Authenticate and issue an access/refresh pair."""
        user = self.authenticate(payload)
        return self.issue_tokens(user)

    def issue_tokens(self, user: User) -> TokenPair:
        """Mint a fresh token pair for ``user``."""
        scopes = ["admin"] if user.is_admin else ["user"]
        access_token = create_access_token(user.id, scopes=scopes, config=self._config)
        refresh_token = create_refresh_token(user.id, scopes=scopes, config=self._config)
        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self._config.access_token_expire_minutes * 60,
            user=UserRead.model_validate(user),
        )

    def refresh(self, refresh_token: str) -> TokenPair:
        """Rotate a refresh token into a new pair."""
        claims = decode_token(
            refresh_token, expected_type="refresh", config=self._config
        )
        user = self._users.get(str(claims["sub"]))
        if user is None:
            raise AuthenticationError(
                "Token subject no longer exists.", code="unknown_subject"
            )
        if not user.is_active:
            raise PermissionDeniedError(
                "This account has been deactivated.", code="account_inactive"
            )
        logger.info("Rotated refresh token for user %s", user.email)
        return self.issue_tokens(user)

    # ----------------------------------------------------------------- profile
    def get_user(self, user_id: str) -> User:
        user = self._users.get(user_id)
        if user is None:
            raise NotFoundError("User not found.", code="user_not_found")
        return user

    def update_profile(self, user: User, payload: UserUpdate) -> User:
        """Update display name and/or password for the current user."""
        if payload.full_name is not None:
            user.full_name = payload.full_name
        if payload.password:
            user.hashed_password = hash_password(payload.password)
        self._session.commit()
        self._session.refresh(user)
        return user

    def deactivate(self, user: User) -> None:
        self._users.deactivate(user)
        self._session.commit()

    def ensure_default_admin(self, email: str, password: str) -> User | None:
        """Idempotently create a bootstrap admin (used on startup in dev)."""
        existing = self._users.get_by_email(email)
        if existing is not None:
            return existing
        user = self._users.create(
            email=email,
            hashed_password=hash_password(password),
            full_name="Administrator",
            role="admin",
        )
        self._session.commit()
        logger.info("Created bootstrap admin account %s", email)
        return user
