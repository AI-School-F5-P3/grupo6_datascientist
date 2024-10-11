from db import Base

from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Integer

class Data(Base):
	__tablename__ = 'data'

	id: Mapped[int] = mapped_column(primary_key=True)
	gender: Mapped[str] = mapped_column(String(50))
	customer: Mapped[str] = mapped_column(String(50))
	age: Mapped[int] = mapped_column(Integer)
	type_travel: Mapped[str] = mapped_column(String(50))
	class_flight: Mapped[str] = mapped_column(String(50))
	distance: Mapped[int] = mapped_column(Integer)
	wifi: Mapped[int] = mapped_column(Integer)
	arrival_time: Mapped[int] = mapped_column(Integer)
	online_booking: Mapped[int] = mapped_column(Integer)
	gate_location: Mapped[int] = mapped_column(Integer)
	food_drink: Mapped[int] = mapped_column(Integer)
	online_boarding: Mapped[int] = mapped_column(Integer)
	seat_confort: Mapped[int] = mapped_column(Integer)
	entertainment: Mapped[int] = mapped_column(Integer)
	on_board: Mapped[int] = mapped_column(Integer)
	leg_room: Mapped[int] = mapped_column(Integer)
	baggage_handling: Mapped[int] = mapped_column(Integer)
	checkin: Mapped[int] = mapped_column(Integer)
	inflight_serv: Mapped[int] = mapped_column(Integer)
	cleanliness: Mapped[int] = mapped_column(Integer)
	departure_delay: Mapped[int] = mapped_column(Integer)
	arrival_dealy: Mapped[int] = mapped_column(Integer)
	prediction: Mapped[str] = mapped_column(String(50))
