#docker exec -it music_backend_ml python update_tag_vectors.py
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.models import Tag

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:Puma1967@db:5432/postgres")
PARQUET_PATH = "app/data/df_unique_tag_embeddings.parquet"


def update_vectors():
    print(f"Łączenie z bazą...")
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text("ALTER TABLE tags_unique ADD COLUMN IF NOT EXISTS tag_embedding vector(768)"))
            conn.commit()

        print(f"Wczytywanie wektorów z: {PARQUET_PATH}")
        if not os.path.exists(PARQUET_PATH):
            print(f"error: Nie znaleziono pliku {PARQUET_PATH}")
            return

        df = pd.read_parquet(PARQUET_PATH)
        print(f"Kolumny w pliku: {df.columns.tolist()}")

        col_name_vec = 'tag_embedding' if 'tag_embedding' in df.columns else 'vector'
        col_name_tag = 'tag' if 'tag' in df.columns else 'name'

        if col_name_vec not in df.columns:
            print(f"error: Nie znaleziono kolumny z wektorem (szukano: {col_name_vec})")
            return

        vector_map = {}
        for idx, row in df.iterrows():
            tag_name = str(row[col_name_tag]).strip().lower()
            vec = row[col_name_vec]
            if hasattr(vec, 'tolist'):
                vec = vec.tolist()
            vector_map[tag_name] = vec

        tags_db = session.query(Tag).all()
        print(f"Znaleziono {len(tags_db)} tagów w bazie.")

        updated_count = 0
        for tag_obj in tags_db:
            t_name = str(tag_obj.name).strip().lower()

            if t_name in vector_map:
                tag_obj.tag_embedding = vector_map[t_name]
                updated_count += 1

            if updated_count > 0 and updated_count % 100 == 0:
                session.commit()
                print(f"Zaktualizowano {updated_count} tagów")

        session.commit()
        print(f"Zaktualizowano {updated_count} z {len(tags_db)} tagów.")

    except Exception as e:
        print(f"error: {e}")
        session.rollback()
    finally:
        session.close()


if __name__ == "__main__":
    update_vectors()