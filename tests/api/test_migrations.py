from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import inspect, text

from legible.api import models  # noqa: F401
from legible.api.db import Base, make_engine
from legible.api.migrate import upgrade


def _diff(engine):
    with engine.connect() as connection:
        context = MigrationContext.configure(connection, opts={"compare_type": True})
        return compare_metadata(context, Base.metadata)


def test_migrations_build_exactly_the_model_schema(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    upgrade(engine)
    upgrade(engine)  # running again is a no-op
    assert _diff(engine) == []
    with engine.connect() as connection:
        assert connection.execute(text("select version_num from alembic_version")).scalar() == "0001"


def test_databases_from_before_migrations_are_adopted(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    Base.metadata.create_all(engine)  # how databases were created before migrations existed
    with engine.begin() as connection:
        connection.execute(text("insert into users (id, email, password_hash, display_name, settings, created_at) "
                                "values ('u1', 'a@b.co', 'x', 'A', '{}', '2026-01-01 00:00:00')"))
    upgrade(engine)
    assert "alembic_version" in inspect(engine).get_table_names()
    with engine.connect() as connection:
        assert connection.execute(text("select count(*) from users")).scalar() == 1
    assert _diff(engine) == []
