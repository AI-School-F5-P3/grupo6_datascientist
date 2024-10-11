from db import Base

from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Integer, Float

class Data(Base):
	__tablename__ = 'data'

	id: Mapped[int] = mapped_column(primary_key=True)
	gender: Mapped[str] = mapped_column(String(10))
	age: Mapped[int] = mapped_column(Integer)
	hypertension: Mapped[int] = mapped_column(Integer)
	heart_disease: Mapped[int] = mapped_column(Integer)
	ever_married: Mapped[str] = mapped_column(String(3))
	work_type: Mapped[str] = mapped_column(String(15))
	residence_type: Mapped[str] = mapped_column(String(15))
	avg_glucose: Mapped[float] = mapped_column(Float)
	bmi: Mapped[float] = mapped_column(Float)
	smoking_status: Mapped[str] = mapped_column(String(25))
	stroke: Mapped[int] = mapped_column(Integer)
