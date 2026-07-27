"""Regression guard: Alembic migrations must recreate the full model schema."""

import os
import sys
from pathlib import Path

import sqlalchemy as sa

PROJECT_ROOT = Path(__file__).parent.parent


def test_upgrade_head_matches_metadata(tmp_path, monkeypatch):
    """`alembic upgrade head` on a fresh DB creates exactly the model tables."""
    db_path = tmp_path / "migration_check.db"
    url = f"sqlite:///{db_path.as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)

    # Fresh interpreter state so db.database picks up the env URL
    for mod in list(sys.modules):
        if mod.startswith(("db.", "config.")) or mod in ("db", "config"):
            sys.modules.pop(mod)

    from alembic import command
    from alembic.config import Config

    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    old_cwd = os.getcwd()
    os.chdir(PROJECT_ROOT)
    try:
        command.upgrade(cfg, "head")
    finally:
        os.chdir(old_cwd)

    from db.models import Base

    inspector = sa.inspect(sa.create_engine(url))
    created = set(inspector.get_table_names()) - {"alembic_version"}
    expected = set(Base.metadata.tables.keys())

    assert expected - created == set(), f"missing tables: {expected - created}"
    assert created - expected == set(), f"unexpected tables: {created - expected}"
