"""Password hashing for the operator accounts. No I/O, no state — pure functions.

`hashlib.scrypt` from the standard library rather than bcrypt or argon2, because this adds
zero dependencies to a project that has deliberately stayed lean, and scrypt is a genuine
memory-hard KDF rather than a fast hash with a salt bolted on.

The honest limit, stated here so nobody has to rediscover it: CPython's scrypt gives back
bytes that we compare with `secrets.compare_digest`, which is constant-time, but the overall
login path is not hardened against a determined attacker enumerating usernames over the
network. For a control-room PoC with a handful of accounts on a LAN that is the right
trade. If this ever faces the public internet with real user accounts, spend one dependency
on `argon2-cffi` and migrate — the stored format below carries its own parameters, so a
migration only needs a new prefix, not a schema change.

Stored format:

    scrypt$n=16384,r=8,p=1$<base64 salt>$<base64 hash>

Parameters live inside the record, so raising the work factor later re-hashes on next login
without invalidating existing accounts or needing a migration table.
"""

import base64
import binascii
import hashlib
import logging
import secrets

logger = logging.getLogger(__name__)

_ALGORITHM = "scrypt"
_SALT_BYTES = 16
_DK_LEN = 32

# r and p are fixed at the usual scrypt defaults; n is the tunable cost. r=8 is what makes
# the memory cost 128 * r * n bytes, which is the figure `_maxmem_for` reproduces below.
_R = 8
_P = 1

# Measured on the dev box (2026-08-10): n=2**14 -> 122 ms, n=2**15 -> 164 ms, n=2**16 -> 333 ms.
# 2**14 sits in the 80-150 ms band that costs an attacker real time without making a guard
# wait at the start of a shift.
DEFAULT_WORK_FACTOR = 14


def _maxmem_for(n: int) -> int:
    """Memory ceiling to hand OpenSSL, in bytes.

    **This must always be passed explicitly.** `hashlib.scrypt` defaults to OpenSSL's 32 MiB
    cap, while scrypt needs `128 * r * n` bytes — 16 MiB at n=2**14 but 32 MiB at n=2**15,
    which trips the cap exactly. Omitting `maxmem` therefore works fine until someone raises
    the work factor one notch, at which point every login raises `ValueError` in production
    and nowhere else. Doubling the requirement leaves headroom for the internal allocation.
    """
    return 128 * _R * n * 2


def hash_password(password: str, work_factor: int = DEFAULT_WORK_FACTOR) -> str:
    """Hash a password with a fresh random salt. Returns the full self-describing record."""
    if not password:
        raise ValueError("Password must not be empty")

    n = 2**work_factor
    salt = secrets.token_bytes(_SALT_BYTES)
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=n,
        r=_R,
        p=_P,
        dklen=_DK_LEN,
        maxmem=_maxmem_for(n),
    )
    return "$".join(
        (
            _ALGORITHM,
            f"n={n},r={_R},p={_P}",
            base64.b64encode(salt).decode("ascii"),
            base64.b64encode(derived).decode("ascii"),
        )
    )


def verify_password(password: str, stored: str) -> bool:
    """Check a password against a stored record. Never raises — a bad record is a failure.

    Fails closed on anything malformed: a truncated file, a hand-edited `users.json`, an
    empty string, a record written by some future algorithm. The alternative is a 500 on the
    login route, which leaks that the account exists and takes the app down for one corrupt
    row rather than denying one login.
    """
    try:
        algorithm, params, salt_b64, hash_b64 = stored.split("$")
        if algorithm != _ALGORITHM:
            logger.warning(f"Unknown password algorithm '{algorithm}'")
            return False

        parsed = dict(pair.split("=", 1) for pair in params.split(","))
        n, r, p = int(parsed["n"]), int(parsed["r"]), int(parsed["p"])
        # n must be a power of two or scrypt itself rejects it; catching it here keeps the
        # failure a denied login rather than a ValueError from OpenSSL.
        if n < 2 or n & (n - 1):
            logger.warning(f"Invalid scrypt cost parameter n={n}")
            return False

        salt = base64.b64decode(salt_b64, validate=True)
        expected = base64.b64decode(hash_b64, validate=True)
        derived = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=n,
            r=r,
            p=p,
            dklen=len(expected),
            maxmem=128 * r * n * 2,
        )
    except (AttributeError, KeyError, ValueError, TypeError, binascii.Error) as exc:
        logger.warning(f"Malformed password record, refusing login: {exc}")
        return False

    return secrets.compare_digest(derived, expected)
