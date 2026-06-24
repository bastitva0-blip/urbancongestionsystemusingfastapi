"""initial schema: postgis, zones, traffic_history (partitioned)

Revision ID: 001
Revises:
Create Date: 2024-08-17 00:00:00.000000

Partitioning strategy decision
───────────────────────────────
We use **native PostgreSQL 16 PARTITION BY RANGE** (monthly partitions) rather
than TimescaleDB hypertables for the following reasons:

1. Zero extra extension dependency — PostGIS is already required; adding
   TimescaleDB would need a custom Docker image or managed-service support.
2. The write pattern is append-mostly with predictable monthly boundaries,
   which maps cleanly to manual range partitions.
3. Dropping an old partition is O(1) metadata-only; no table lock on the parent.
4. TimescaleDB would add automated compression and retention policies, but those
   features aren't needed at this stage and can always be layered on later.

Trade-off accepted: we manually create each monthly child partition. A cron job
or second migration handles future months. TimescaleDB would automate this.
"""

from alembic import op
import sqlalchemy as sa
import geoalchemy2

revision = "001"
down_revision = None
branch_labels = None
depends_on = None

# Pre-create partitions for a rolling 6-month window
INITIAL_PARTITIONS = [
    ("2024_07", "2024-07-01", "2024-08-01"),
    ("2024_08", "2024-08-01", "2024-09-01"),
    ("2024_09", "2024-09-01", "2024-10-01"),
    ("2024_10", "2024-10-01", "2024-11-01"),
    ("2024_11", "2024-11-01", "2024-12-01"),
    ("2024_12", "2024-12-01", "2025-01-01"),
    ("2025_01", "2025-01-01", "2025-02-01"),
    ("2025_02", "2025-02-01", "2025-03-01"),
    ("2025_03", "2025-03-01", "2025-04-01"),
    ("2025_04", "2025-04-01", "2025-05-01"),
    ("2025_05", "2025-05-01", "2025-06-01"),
    ("2025_06", "2025-06-01", "2025-07-01"),
]


def upgrade() -> None:
    # 1. PostGIS extension
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")

    # 2. zones table
    op.create_table(
        "zones",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("zone_name", sa.String(120), nullable=False),
        sa.Column(
            "boundary",
            geoalchemy2.types.Geometry(geometry_type="POLYGON", srid=4326),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_zones"),
        sa.UniqueConstraint("zone_name", name="uq_zones_name"),
    )
    op.create_index("ix_zones_boundary", "zones", ["boundary"], postgresql_using="gist")

    # 3. Partitioned parent table (no rows stored here directly)
    op.execute("""
        CREATE TABLE traffic_history (
            id          BIGSERIAL       NOT NULL,
            reported_at TIMESTAMPTZ     NOT NULL,
            zone_id     INTEGER         NOT NULL REFERENCES zones(id) ON DELETE RESTRICT,
            vehicle_count   INTEGER,
            average_speed   FLOAT,
            congestion_level FLOAT      NOT NULL,
            CONSTRAINT pk_traffic_history PRIMARY KEY (id, reported_at)
        ) PARTITION BY RANGE (reported_at)
    """)

    op.create_index(
        "ix_traffic_zone_time",
        "traffic_history",
        ["zone_id", "reported_at"],
        postgresql_using="btree",
    )

    # 4. Monthly child partitions
    for suffix, start, end in INITIAL_PARTITIONS:
        op.execute(f"""
            CREATE TABLE traffic_history_{suffix}
            PARTITION OF traffic_history
            FOR VALUES FROM ('{start}') TO ('{end}')
        """)

    # 5. Seed a default "unknown" zone so zone_id=0 writes don't FK-fail
    # (In production you'd seed real zone polygons via a separate data migration)
    op.execute("""
        INSERT INTO zones (zone_name, boundary)
        VALUES (
            'downtown',
            ST_GeomFromText(
                'POLYGON((-74.02 40.70, -73.97 40.70, -73.97 40.75, -74.02 40.75, -74.02 40.70))',
                4326
            )
        )
    """)


def downgrade() -> None:
    for suffix, _, _ in INITIAL_PARTITIONS:
        op.execute(f"DROP TABLE IF EXISTS traffic_history_{suffix}")
    op.execute("DROP TABLE IF EXISTS traffic_history")
    op.drop_table("zones")
    op.execute("DROP EXTENSION IF EXISTS postgis CASCADE")
