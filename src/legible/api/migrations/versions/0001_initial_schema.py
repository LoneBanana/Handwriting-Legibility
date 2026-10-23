"""initial schema

Revision ID: 0001
Revises: 
Create Date: 2026-10-23 10:26:47.765704
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('users',
    sa.Column('id', sa.String(length=32), nullable=False),
    sa.Column('email', sa.String(length=320), nullable=False),
    sa.Column('password_hash', sa.String(length=255), nullable=False),
    sa.Column('display_name', sa.String(length=80), nullable=False),
    sa.Column('settings', sa.JSON(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_users_email'), ['email'], unique=True)

    op.create_table('converse_sessions',
    sa.Column('id', sa.String(length=32), nullable=False),
    sa.Column('user_id', sa.String(length=32), nullable=False),
    sa.Column('provider', sa.String(length=16), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('ended_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('settings', sa.JSON(), nullable=False),
    sa.Column('tavus_conversation_id', sa.String(length=128), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('converse_sessions', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_converse_sessions_user_id'), ['user_id'], unique=False)

    op.create_table('fact_memory',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('user_id', sa.String(length=32), nullable=False),
    sa.Column('fact', sa.String(length=64), nullable=False),
    sa.Column('skill', sa.String(length=64), nullable=False),
    sa.Column('seen', sa.Integer(), nullable=False),
    sa.Column('correct', sa.Integer(), nullable=False),
    sa.Column('fluent', sa.Integer(), nullable=False),
    sa.Column('mean_latency_ms', sa.Float(), nullable=True),
    sa.Column('last_seen_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id', 'fact')
    )
    with op.batch_alter_table('fact_memory', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_fact_memory_user_id'), ['user_id'], unique=False)

    op.create_table('handwriting_samples',
    sa.Column('id', sa.String(length=32), nullable=False),
    sa.Column('user_id', sa.String(length=32), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('image_path', sa.String(length=512), nullable=False),
    sa.Column('lines', sa.JSON(), nullable=False),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('model_version', sa.String(length=64), nullable=False),
    sa.Column('corrected_text', sa.Text(), nullable=True),
    sa.Column('corrected_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('used_in_adapter', sa.Boolean(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('handwriting_samples', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_handwriting_samples_user_id'), ['user_id'], unique=False)

    op.create_table('personal_adapters',
    sa.Column('user_id', sa.String(length=32), nullable=False),
    sa.Column('path', sa.String(length=512), nullable=False),
    sa.Column('base_model_version', sa.String(length=64), nullable=False),
    sa.Column('samples', sa.Integer(), nullable=False),
    sa.Column('lines', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('user_id')
    )
    op.create_table('practice_sessions',
    sa.Column('id', sa.String(length=32), nullable=False),
    sa.Column('user_id', sa.String(length=32), nullable=False),
    sa.Column('kind', sa.String(length=16), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('payload', sa.JSON(), nullable=False),
    sa.Column('responses', sa.JSON(), nullable=False),
    sa.Column('result', sa.JSON(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('practice_sessions', schema=None) as batch_op:
        batch_op.create_index('ix_practice_sessions_user_kind', ['user_id', 'kind'], unique=False)

    op.create_table('track_events',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('user_id', sa.String(length=32), nullable=False),
    sa.Column('track_key', sa.String(length=64), nullable=False),
    sa.Column('local_date', sa.Date(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('score', sa.Float(), nullable=False),
    sa.Column('perfect', sa.Boolean(), nullable=False),
    sa.Column('level_before', sa.Integer(), nullable=False),
    sa.Column('level_after', sa.Integer(), nullable=False),
    sa.Column('reason', sa.String(length=16), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('track_events', schema=None) as batch_op:
        batch_op.create_index('ix_track_events_user_track_date', ['user_id', 'track_key', 'local_date'], unique=False)

    op.create_table('track_states',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('user_id', sa.String(length=32), nullable=False),
    sa.Column('track_key', sa.String(length=64), nullable=False),
    sa.Column('level', sa.Integer(), nullable=False),
    sa.Column('streak', sa.Integer(), nullable=False),
    sa.Column('best_streak', sa.Integer(), nullable=False),
    sa.Column('sessions', sa.Integer(), nullable=False),
    sa.Column('perfect_sessions', sa.Integer(), nullable=False),
    sa.Column('struggle_count', sa.Integer(), nullable=False),
    sa.Column('last_promoted_on', sa.Date(), nullable=True),
    sa.Column('promotions_on_last_date', sa.Integer(), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id', 'track_key')
    )
    with op.batch_alter_table('track_states', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_track_states_user_id'), ['user_id'], unique=False)

    op.create_table('word_memory',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('user_id', sa.String(length=32), nullable=False),
    sa.Column('word', sa.String(length=64), nullable=False),
    sa.Column('mode', sa.String(length=8), nullable=False),
    sa.Column('skill', sa.String(length=64), nullable=False),
    sa.Column('seen', sa.Integer(), nullable=False),
    sa.Column('correct', sa.Integer(), nullable=False),
    sa.Column('wrong', sa.Integer(), nullable=False),
    sa.Column('half_life_days', sa.Float(), nullable=False),
    sa.Column('last_seen_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id', 'word', 'mode')
    )
    with op.batch_alter_table('word_memory', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_word_memory_user_id'), ['user_id'], unique=False)

    op.create_table('converse_turns',
    sa.Column('id', sa.String(length=32), nullable=False),
    sa.Column('session_id', sa.String(length=32), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('user_text', sa.Text(), nullable=False),
    sa.Column('assistant_text', sa.Text(), nullable=False),
    sa.Column('segments', sa.JSON(), nullable=False),
    sa.Column('complete', sa.Boolean(), nullable=False),
    sa.ForeignKeyConstraint(['session_id'], ['converse_sessions.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('converse_turns', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_converse_turns_session_id'), ['session_id'], unique=False)

    op.create_table('comprehension_checks',
    sa.Column('id', sa.String(length=32), nullable=False),
    sa.Column('user_id', sa.String(length=32), nullable=False),
    sa.Column('turn_id', sa.String(length=32), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('questions', sa.JSON(), nullable=False),
    sa.Column('submitted_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('result', sa.JSON(), nullable=True),
    sa.ForeignKeyConstraint(['turn_id'], ['converse_turns.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('comprehension_checks', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_comprehension_checks_user_id'), ['user_id'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('comprehension_checks', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_comprehension_checks_user_id'))

    op.drop_table('comprehension_checks')
    with op.batch_alter_table('converse_turns', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_converse_turns_session_id'))

    op.drop_table('converse_turns')
    with op.batch_alter_table('word_memory', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_word_memory_user_id'))

    op.drop_table('word_memory')
    with op.batch_alter_table('track_states', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_track_states_user_id'))

    op.drop_table('track_states')
    with op.batch_alter_table('track_events', schema=None) as batch_op:
        batch_op.drop_index('ix_track_events_user_track_date')

    op.drop_table('track_events')
    with op.batch_alter_table('practice_sessions', schema=None) as batch_op:
        batch_op.drop_index('ix_practice_sessions_user_kind')

    op.drop_table('practice_sessions')
    op.drop_table('personal_adapters')
    with op.batch_alter_table('handwriting_samples', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_handwriting_samples_user_id'))

    op.drop_table('handwriting_samples')
    with op.batch_alter_table('fact_memory', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_fact_memory_user_id'))

    op.drop_table('fact_memory')
    with op.batch_alter_table('converse_sessions', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_converse_sessions_user_id'))

    op.drop_table('converse_sessions')
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_users_email'))

    op.drop_table('users')
