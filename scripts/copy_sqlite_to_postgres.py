"""Copy every row from the local SQLite snapshot into a Postgres database.

Usage:
    DATABASE_URL=postgresql://... python scripts/copy_sqlite_to_postgres.py [--source data/brandclave.db] [--truncate]

Run `alembic upgrade head` against the same DATABASE_URL first so the tables
exist. Tables are copied in dependency order; --truncate empties the target
tables first so the copy is idempotent. Verified by comparing row counts.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine, insert, select, text, func  # noqa: E402

from db.database import normalize_database_url  # noqa: E402
from db.models import Base  # noqa: E402

BATCH = 500


def ordered_tables():
    """Parents before children (SQLAlchemy sorts by foreign keys)."""
    return list(Base.metadata.sorted_tables)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/brandclave.db")
    parser.add_argument("--target", default=os.environ.get("DATABASE_URL", ""))
    parser.add_argument("--truncate", action="store_true")
    args = parser.parse_args()
    if not args.target or args.target.startswith("sqlite"):
        print("set DATABASE_URL (or --target) to the Postgres URL", file=sys.stderr)
        return 2

    src = create_engine(f"sqlite:///{args.source}")
    dst = create_engine(normalize_database_url(args.target), pool_pre_ping=True)
    tables = ordered_tables()

    with dst.begin() as conn:
        if args.truncate:
            for table in reversed(tables):
                conn.execute(text(f'TRUNCATE TABLE "{table.name}" CASCADE'))
            print("target tables emptied")

    problems = 0
    with src.connect() as sconn:
        for table in tables:
            rows = [dict(r._mapping) for r in sconn.execute(select(table)).fetchall()]
            with dst.begin() as dconn:
                for i in range(0, len(rows), BATCH):
                    dconn.execute(insert(table), rows[i: i + BATCH])
                copied = dconn.execute(select(func.count()).select_from(table)).scalar()
            flag = "" if copied == len(rows) else "  <-- MISMATCH"
            if flag:
                problems += 1
            print(f"{table.name:20s} {len(rows):6d} -> {copied:6d}{flag}")

    # alembic_version is not in the model metadata; the target got it from `alembic upgrade head`.
    with dst.connect() as conn:
        version = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
    print("alembic version on target:", version)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
