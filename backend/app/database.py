# app/database.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()  # wczytaj .env z katalogu projektu


#Pobiera adres bazy danych ze zmiennych i jeśli go nie znajdzie, natychmiast wyłącza program z błędem
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("Brakuje DATABASE_URL w .env")




#łącze między kodem Python a bazą PostgreSQL,
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

#szablon, który tworzy połączenie z bazą
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


Base = declarative_base()


##python -m uvicorn app.main:app --reload