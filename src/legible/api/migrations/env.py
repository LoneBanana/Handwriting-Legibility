"""Alembic environment: runs on a connection handed over by legible.api.migrate, or
connects to the configured database (``LEGIBLE_DATABASE_URL``) when run by ``alembic``."""

from __future__ import annotations

from alembic import context

from legible.api import models  # noqa: F401  (registers every table on Base.metadata)
from legible.api.db import Base, make_engine
from legible.api.settings import get_settings

config = context.config
target_metadata = Base.metadata


def _configure(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=connection.dialect.name == "sqlite",  # SQLite can't ALTER most things
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_offline() -> None:
    context.configure(url=config.get_main_option("sqlalchemy.url") or get_settings().resolved_database_url,
                      target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        _configure(connection)
        return
    engine = make_engine(config.get_main_option("sqlalchemy.url") or get_settings().resolved_database_url)
    try:
        with engine.connect() as connection:
            _configure(connection)
            connection.commit()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_offline()
else:
    run_online()
