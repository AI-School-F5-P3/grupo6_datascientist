from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import joblib
import pandas as pd

import model

with open('./model_pipeline.pkl', 'rb') as f:
    pipeline = joblib.load('./model_pipeline.pkl')

def post_data(db: Session, datax: model.FeatureSchema):

	x_dict = datax.model_dump()

	cols = ['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class',
       'Flight Distance', 'Inflight wifi service',
       'Departure/Arrival time convenient', 'Ease of Online booking',
       'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
       'Inflight entertainment', 'On-board service', 'Leg room service',
       'Baggage handling', 'Checkin service', 'Inflight service',
       'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes',
       'Age Group']
	
	df = pd.DataFrame([list(x_dict.values())], columns=cols)
	df['Age Group'] = df['Age Group'].astype('category')
	
	try:
		prediction = pipeline.predict(df)
		data_db = datax.model_dump()
		data_db['prediction'] = int(prediction[0])
		new_data = model.Data(**data_db)
		db.add(new_data)
		db.commit()
		return {"pred": data_db['prediction']}
	except SQLAlchemyError:
		db.rollback()
	except Exception:
		db.rollback()
	return None
	
def get_all_data(db: Session):
	pass