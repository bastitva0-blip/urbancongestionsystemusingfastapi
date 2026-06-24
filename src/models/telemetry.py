from datetime import datetime
from sqlalchemy import (
    BigInteger,
    Float,
    ForeignKey,
    Index,
    Integer,
    PrimaryKeyConstraint,
    TIMESTAMP,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.db.base import Base


class TrafficHistory(Base):
    """
    Partitioned by RANGE on reported_at (monthly).

    PostgreSQL requires the partition key to be part of the primary key on
    natively-partitioned tables. Hence the composite PK (id, reported_at).
    id is still a BIGSERIAL — uniqueness is enforced at the application level
    and via the sequence; Postgres propagates it across partitions.
    """

    __tablename__ = "traffic_history"
    __table_args__ = (
        # Composite PK satisfies Postgres partitioning constraint
        PrimaryKeyConstraint("id", "reported_at", name="pk_traffic_history"),
        # Optimised lookup: filter by zone then time-range
        Index("ix_traffic_zone_time", "zone_id", "reported_at"),
        # Partition declaration is handled in migration SQL — not here
        {"postgresql_partition_by": "RANGE (reported_at)"},
    )

    id: Mapped[int] = mapped_column(BigInteger, autoincrement=True, nullable=False)
    reported_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    zone_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("zones.id", ondelete="RESTRICT"), nullable=False
    )
    vehicle_count: Mapped[int] = mapped_column(Integer, nullable=True)
    average_speed: Mapped[float] = mapped_column(Float, nullable=True)
    congestion_level: Mapped[float] = mapped_column(Float, nullable=False)

    zone: Mapped["Zone"] = relationship("Zone", back_populates="history", lazy="noload")

    def __repr__(self) -> str:
        return (
            f"<TrafficHistory id={self.id} zone={self.zone_id} "
            f"congestion={self.congestion_level:.2f}>"
        )
