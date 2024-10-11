import dotenv
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Load environment variables from .env file
dotenv.load_dotenv()

# Fetch the environment variables
USER = os.getenv('MY_USER')
PASS = os.getenv('MY_PASSWORD')
HOST = os.getenv('MY_HOST')
PORT = os.getenv('MY_PORT')
DB = os.getenv('MY_DB')

# Correct the f-string to use the right variable names
DATABASE_URL = f"mysql+mysqlconnector://{USER}:{PASS}@{HOST}:{PORT}/{DB}"

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create session and base
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()
