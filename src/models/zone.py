from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from geoalchemy2 import Geometry
from src.db.base import Base


class Zone(Base):
    __tablename__ = "zones"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    zone_name: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)
    # EPSG:4326 — standard lat/lng WGS84
    boundary: Mapped[bytes] = mapped_column(
        Geometry(geometry_type="POLYGON", srid=4326), nullable=False
    )

    history: Mapped[list["TrafficHistory"]] = relationship(
        "TrafficHistory", back_populates="zone", lazy="noload"
    )

    def __repr__(self) -> str:
        return f"<Zone id={self.id} name={self.zone_name!r}>"
