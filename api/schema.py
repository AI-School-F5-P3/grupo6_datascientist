from pydantic import BaseModel

class FeatureSchema(BaseModel):
	gender: str
	customer: str
	age: int
	type_travel: str
	class_flight: str
	distance: int
	wifi: int
	arrival_time: int
	online_booking: int
	gate_location: int
	food_drink: int
	online_boarding: int
	seat_confort: int
	entertainment: int
	on_board: int
	leg_room: int
	baggage_handling: int
	checkin: int
	inflight_serv: int
	cleanliness: int
	departure_delay: int
	arrival_dealy: int
