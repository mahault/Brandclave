"""First-boot bootstrap for a managed Postgres.

When DATABASE_URL points at Postgres, the web process runs the Alembic
migrations at startup and, if the database holds no content yet, copies the
committed SQLite snapshot (data/brandclave.db) into it. Both steps are
idempotent: migrations are versioned, and the copy only runs into an empty
database. This lets the live service move from the ephemeral file to a
persistent database with nothing but an environment variable.
"""

import logging
from pathlib import Path

from sqlalchemy import create_engine, func, insert, select, text

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = REPO_ROOT / "data" / "brandclave.db"
BATCH = 500


def _run_migrations() -> None:
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    command.upgrade(cfg, "head")
    logger.info("Alembic migrations applied")


def _copy_snapshot(target_engine) -> dict[str, int]:
    from db.models import Base

    if not SNAPSHOT.exists():
        logger.warning(f"No SQLite snapshot at {SNAPSHOT}; database starts empty")
        return {}
    source = create_engine(f"sqlite:///{SNAPSHOT.as_posix()}")
    copied: dict[str, int] = {}
    with source.connect() as sconn:
        for table in Base.metadata.sorted_tables:
            rows = [dict(r._mapping) for r in sconn.execute(select(table)).fetchall()]
            if not rows:
                copied[table.name] = 0
                continue
            with target_engine.begin() as tconn:
                for i in range(0, len(rows), BATCH):
                    tconn.execute(insert(table), rows[i: i + BATCH])
            copied[table.name] = len(rows)
    return copied


def _add_missing_blueprints(target_engine) -> int:
    """Blueprints are demo content shipped in the snapshot; a database seeded
    from an older snapshot should still pick up the ones added since."""
    from db.models import BrandBlueprintModel

    if not SNAPSHOT.exists():
        return 0
    table = BrandBlueprintModel.__table__
    source = create_engine(f"sqlite:///{SNAPSHOT.as_posix()}")
    with source.connect() as sconn:
        rows = [dict(r._mapping) for r in sconn.execute(select(table)).fetchall()]
    with target_engine.begin() as tconn:
        have = {r[0] for r in tconn.execute(select(table.c.id)).fetchall()}
        missing = [r for r in rows if r["id"] not in have]
        if missing:
            tconn.execute(insert(table), missing)
    return len(missing)


def bootstrap_if_postgres(database_url: str) -> None:
    """Migrate, then seed from the snapshot when the database is empty."""
    if not database_url.startswith("postgresql"):
        return
    from db.models import RawContentModel

    _run_migrations()
    engine = create_engine(database_url, pool_pre_ping=True)
    with engine.connect() as conn:
        existing = conn.execute(select(func.count()).select_from(RawContentModel.__table__)).scalar() or 0
    if existing:
        logger.info(f"Postgres already holds {existing} content rows; no full seed")
        added = _add_missing_blueprints(engine)
        if added:
            logger.info(f"Added {added} blueprint(s) present in the snapshot but not in Postgres")
        return
    logger.info("Postgres is empty; seeding from the committed SQLite snapshot")
    copied = _copy_snapshot(engine)
    logger.info("Seeded: " + ", ".join(f"{k}={v}" for k, v in copied.items() if v))
    with engine.connect() as conn:
        # sequences are not used (string ids) but keep the check explicit
        conn.execute(text("SELECT 1"))
