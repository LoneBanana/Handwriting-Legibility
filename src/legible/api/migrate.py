"""Database migrations (Alembic), applied automatically when the API starts.

    legible-db upgrade                      # apply pending migrations to LEGIBLE_DATABASE_URL
    legible-db current                      # show the database's revision
    legible-db revision -m "add x column"   # after changing models.py: autogenerate a migration

Databases created before migrations existed (by ``create_all``) match the first
revision, so they are stamped with it and then upgraded normally.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect

from .db import make_engine
from .settings import get_settings

INITIAL_REVISION = "0001"


def alembic_config() -> Config:
    config = Config()
    config.set_main_option("script_location", str(Path(__file__).with_name("migrations")))
    return config


def upgrade(engine: Engine, revision: str = "head") -> None:
    config = alembic_config()
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        tables = set(inspect(connection).get_table_names())
        if "users" in tables and "alembic_version" not in tables:
            command.stamp(config, INITIAL_REVISION)
        command.upgrade(config, revision)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage the Legible database schema.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("upgrade", help="apply pending migrations")
    sub.add_parser("current", help="show the current revision")
    rev = sub.add_parser("revision", help="autogenerate a migration from model changes")
    rev.add_argument("-m", "--message", required=True)
    args = parser.parse_args(argv)

    engine = make_engine(get_settings().resolved_database_url)
    try:
        if args.command == "upgrade":
            upgrade(engine)
            print("database is up to date")
            return 0
        config = alembic_config()
        with engine.begin() as connection:
            config.attributes["connection"] = connection
            if args.command == "current":
                command.current(config, verbose=True)
            else:
                command.revision(config, message=args.message, autogenerate=True)
    finally:
        engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(main())
