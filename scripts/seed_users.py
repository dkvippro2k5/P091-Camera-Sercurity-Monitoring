"""Create and manage the operator accounts in `data/users.json`.

    python scripts/seed_users.py                          # interactive: create the first two
    python scripts/seed_users.py --add --role guard       # add one more
    python scripts/seed_users.py --list                   # who exists (never shows hashes)

There is no signup endpoint by design — see `src/services/user_store.py`. Accounts are made
here, on the box, by whoever runs the shift.

Passwords are always prompted for, never taken as an argument: anything on argv lands in
shell history and is visible in the process list to every other user on the machine.
"""

import argparse
import getpass
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.services.user_store import DEFAULT_USERS_PATH, UserStore  # noqa: E402

# Short enough not to encourage a sticky note, long enough to be worth hashing.
_MIN_PASSWORD_LENGTH = 8


def _prompt_password(username: str) -> str | None:
    """Ask twice and compare. Returns None if the operator gives up."""
    for _ in range(3):
        first = getpass.getpass(f"  Password for '{username}': ")
        if len(first) < _MIN_PASSWORD_LENGTH:
            print(f"  ! At least {_MIN_PASSWORD_LENGTH} characters, please.")
            continue
        if first != getpass.getpass("  Confirm: "):
            print("  ! They do not match.")
            continue
        return first
    print("  ! Giving up after 3 attempts.")
    return None


def _create(store: UserStore, username: str, role: str, display_name: str) -> bool:
    password = _prompt_password(username)
    if password is None:
        return False
    try:
        store.add_user(username=username, password=password, role=role, display_name=display_name)
    except ValueError as exc:
        print(f"  ! {exc}")
        return False
    print(f"  ✓ {role} '{username}' created")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage P-091 operator accounts")
    parser.add_argument("--add", action="store_true", help="Add one account instead of seeding both")
    parser.add_argument("--list", action="store_true", help="List existing accounts")
    parser.add_argument("--role", choices=("guard", "manager"), help="Role for --add")
    parser.add_argument("--username", help="Username for --add")
    parser.add_argument("--path", default=DEFAULT_USERS_PATH, help="Where users.json lives")
    args = parser.parse_args()

    store = UserStore(path=args.path)

    if args.list:
        users = store.list_users()
        if not users:
            print(f"No accounts in {args.path}. Run this script with no arguments to create them.")
            return 1
        print(f"{len(users)} account(s) in {args.path}:\n")
        for user in sorted(users, key=lambda u: (u.role, u.username)):
            print(f"  {user.role:<8}  {user.username:<16}  {user.display_name}")
        return 0

    if args.add:
        username = args.username or input("Username: ").strip()
        role = args.role or input("Role (guard/manager): ").strip()
        if role not in ("guard", "manager"):
            print(f"! '{role}' is not a role. Use guard or manager.")
            return 1
        display_name = input(f"Display name [{username}]: ").strip() or username
        return 0 if _create(store, username, role, display_name) else 1

    # Default: seed one of each, which is the minimum for the app to be usable.
    if store.list_users():
        print(f"{args.path} already has accounts. Use --add for another, or --list to see them.")
        return 1

    print("Creating the first two accounts.\n")
    print("  A GUARD runs the room: the live grid, the alerts, and the camera setup.")
    print("  A MANAGER reads the statistics report and nothing else.\n")

    for role in ("manager", "guard"):
        default = role
        username = input(f"{role.capitalize()} username [{default}]: ").strip() or default
        display_name = input(f"  Display name [{username}]: ").strip() or username
        if not _create(store, username, role, display_name):
            return 1

    print(f"\nDone — {args.path}")
    print("Passwords are stored as scrypt hashes; this file is gitignored. Keep it that way.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
