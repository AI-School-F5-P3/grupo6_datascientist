from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Define the DATABASE_URL
DATABASE_URL = 'mysql+pymysql://admindb:grupo_06@ictusdb.mysql.database.azure.com/ictusdb?charset=utf8mb4'

# Create engine and base
engine = create_engine(DATABASE_URL)
Base = declarative_base()

# Define the Patient model
class Patient(Base):
    __tablename__ = 'patients'
    
    id = Column(Integer, primary_key=True)
    gender = Column(String(10))
    age = Column(Integer)
    hypertension = Column(Integer)
    heart_disease = Column(Integer)
    ever_married = Column(String(5))
    work_type = Column(String(20))
    residence_type = Column(String(10))
    avg_glucose_level = Column(Float)
    bmi = Column(Float)
    smoking_status = Column(String(20))
    stroke = Column(Integer)

    def __repr__(self):
        return f"<Patient(id={self.id}, age={self.age}, stroke={self.stroke})>"

# Create session factory
Session = sessionmaker(bind=engine)

def initialize_database():
    """Initialize database by creating all tables if they don't exist."""
    try:
        Base.metadata.create_all(engine)
        logging.info("Database initialized successfully")
        return True
    except Exception as e:
        logging.error(f"Error initializing database: {str(e)}")
        return False