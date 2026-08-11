"""Operator accounts, read from `data/users.json`.

Two roles, mapped straight onto who does what in a control room:

- **guard** — runs the room: watches the live grid, judges alerts real or false, and
  configures the cameras (add / delete / upload / draw zones).
- **manager** — reads the statistics report. Nothing else is offered to them.

There is no signup endpoint on purpose. A 24/7 monitoring station has a known, small roster
set by whoever runs the shift, and a public registration route on a system that can add
cameras and read incident footage is a liability with no matching benefit. Accounts are
created by `scripts/seed_users.py`, which prompts for the password rather than taking it on
argv (argv lands in shell history and in the process list).

`data/users.json` lives under the gitignored `data/` next to `cameras.json`, so hashes never
reach the repository.
"""

import json
import logging
import os
import threading
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel

from src.services.security import DEFAULT_WORK_FACTOR, hash_password, verify_password

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_USERS_PATH = os.path.join(_PROJECT_ROOT, "data", "users.json")

Role = Literal["guard", "manager"]


class User(BaseModel):
    """An account as the rest of the app sees it — no hash, ever.

    `UserRecord` below is the on-disk shape and is the only thing that carries the hash.
    Keeping them as two types is what makes "the hash never leaves the server" a property of
    the type system rather than a rule someone has to remember at each endpoint.
    """

    username: str
    role: Role
    display_name: str


class UserRecord(User):
    """The stored form: a `User` plus its password hash and enabled flag."""

    password_hash: str
    active: bool = True

    def public(self) -> User:
        return User(username=self.username, role=self.role, display_name=self.display_name)


class UserStore:
    """Loads accounts once, verifies passwords, and can append a new account.

    Held as a process singleton on `app.state`. The file is small (a roster, not a table), so
    it is read whole and kept in memory; a write rewrites it atomically.
    """

    def __init__(self, path: str = DEFAULT_USERS_PATH) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._users: dict[str, UserRecord] = {}
        self.reload()

    def reload(self) -> None:
        """Re-read the file. Missing or corrupt means "no accounts", never a crash.

        A malformed roster must not stop the app from booting — the alert pipeline is the
        thing that matters most and it should keep running while somebody fixes the file.
        Every login then fails closed, which is loud enough to be noticed immediately.
        """
        with self._lock:
            self._users = {}
            if not os.path.exists(self.path):
                logger.warning(f"No user file at {self.path}; run scripts/seed_users.py to create accounts")
                return
            try:
                with open(self.path, encoding="utf-8") as handle:
                    raw = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                logger.error(f"Could not read {self.path}, no accounts will authenticate: {exc}")
                return

            for entry in raw.get("users", []):
                try:
                    record = UserRecord(**entry)
                except (TypeError, ValueError) as exc:
                    logger.error(f"Skipping malformed user entry: {exc}")
                    continue
                self._users[record.username.lower()] = record
            logger.info(f"Loaded {len(self._users)} account(s) from {self.path}")

    def authenticate(self, username: str, password: str) -> User | None:
        """Verify a login. Returns the public user on success, None on any failure.

        One indistinguishable failure for "no such user", "wrong password", and "account
        disabled": the login route must not let an attacker enumerate valid usernames by
        watching for a different error. The dummy verify on the unknown-user path is what
        keeps the timing from doing the same thing the error message would have.
        """
        with self._lock:
            record = self._users.get(username.strip().lower())

        if record is None:
            # Burn comparable time so a missing account is not measurably faster than a wrong
            # password. Not perfect, but it removes the trivially observable difference.
            verify_password(password, _dummy_hash())
            return None

        if not verify_password(password, record.password_hash):
            logger.info(f"Failed login for '{record.username}'")
            return None

        if not record.active:
            logger.info(f"Login refused for disabled account '{record.username}'")
            return None

        return record.public()

    def get(self, username: str) -> User | None:
        """Look up an account by name. Used to re-check the role behind a live session.

        Resolved per request rather than trusted from the session, so demoting a manager or
        disabling an account takes effect on their next request instead of at the end of a
        12-hour shift.
        """
        with self._lock:
            record = self._users.get(username.strip().lower())
        if record is None or not record.active:
            return None
        return record.public()

    def list_users(self) -> list[User]:
        with self._lock:
            return [record.public() for record in self._users.values()]

    def add_user(
        self,
        username: str,
        password: str,
        role: Role,
        display_name: str = "",
        work_factor: int = DEFAULT_WORK_FACTOR,
    ) -> User:
        """Create an account and persist it. Raises if the username is taken.

        `work_factor` is exposed so tests can hash at a cheap cost. It is safe to vary per
        account: `security.py` stores the parameters inside each record, so a hash made at one
        factor still verifies at that factor after the default moves.
        """
        clean = username.strip()
        if not clean:
            raise ValueError("Username must not be empty")

        record = UserRecord(
            username=clean,
            role=role,
            display_name=display_name.strip() or clean,
            password_hash=hash_password(password, work_factor=work_factor),
        )
        with self._lock:
            if clean.lower() in self._users:
                raise ValueError(f"User '{clean}' already exists")
            self._users[clean.lower()] = record
            self._write_locked()
        logger.info(f"Created {role} account '{clean}'")
        return record.public()

    def _write_locked(self) -> None:
        """Rewrite the roster atomically. Caller must hold the lock.

        Temp file plus `os.replace` (atomic on NTFS and POSIX alike) so a crash mid-write
        cannot leave a half-written roster that locks everyone out.
        """
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        payload = {"users": [record.model_dump() for record in self._users.values()]}
        temp_path = f"{self.path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temp_path, self.path)


@lru_cache(maxsize=1)
def _dummy_hash() -> str:
    """A real hash to verify against when the username does not exist.

    Computed on first use rather than at import, so the work factor's cost is not added to
    every process start (including test collection) for a value only needed when someone
    tries to log in as a user that does not exist.
    """
    return hash_password("dummy-password-for-constant-time-path")
