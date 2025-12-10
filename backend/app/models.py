# app/models.py
from sqlalchemy import Column, Integer, String, Boolean, Float, JSON
from .database import Base

class Song(Base):
    __tablename__ = "spotify_tracks"  # Nazwa tabeli w Postgres

    track_id = Column(String, primary_key=True)
    name = Column(String)
    artist = Column(String)
    spotify_preview_url = Column(String)
    tags = Column(String)
    genre = Column(String)
    year = Column(Integer)
    duration_ms = Column(Integer)
    danceability = Column(Float)
    energy = Column(Float)
    key = Column(Integer)
    loudness = Column(Float)
    mode = Column(Integer)
    speechiness = Column(Float)
    acousticness = Column(Float)
    instrumentalness = Column(Float)
    liveness = Column(Float)
    valence = Column(Float)
    tempo = Column(Float)
    time_signature = Column(Integer)
    album_name = Column(String)
    popularity = Column(Float)
    spotify_url = Column(String)
    explicit = Column(Boolean)
    album_images = Column(JSON)
    spotify_id = Column(String)
    n_tempo = Column(Float)
    n_loudness = Column(Float)
    tags_list = Column(String)
    tags_count = Column(Integer)
