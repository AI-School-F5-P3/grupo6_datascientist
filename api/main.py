from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

from db import engine, SessionLocal
import model
import crud

app = FastAPI()

model.Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/predict/")
def predict(data: model.FeatureSchema, db: Session = Depends(get_db)):
    result = crud.post_data(db, data)
    if result:
        # Suponiendo que 'result' contiene una predicción de si el cliente estará satisfecho
        return {"msg": "ok", "data": data, "prediction": result['pred']}
    else:
        return {"msg": "error"}
