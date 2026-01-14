# app/database.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

# pg_dump -U postgres -d postgres > init-db/init.sql
# docker exec -t promptify-db-1  pg_dump -U postgres -d postgres --clean --if-exists > init-db/init.sql

load_dotenv()


DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("Brakuje DATABASE_URL w .env")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

##python -m uvicorn app.main:app --reload