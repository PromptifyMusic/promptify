# app/models.py
from sqlalchemy import Column, Integer, String, Boolean, Float, Text, ForeignKey, Table
from sqlalchemy.orm import relationship, declarative_base


Base = declarative_base()



# ...
# 1. TABELA MAPUJĄCA (ASSOCIATION TABLE)
song_tag_association = Table(
    'song_tags_map', Base.metadata,
    Column('song_id', String, ForeignKey('songs_master.song_id'), primary_key=True),
    Column('tag_id', Integer, ForeignKey('tags_unique.tag_id'), primary_key=True)
)


class Tag(Base):
    __tablename__ = "tags_unique"

    tag_id = Column(Integer, primary_key=True)
    name = Column(String)

    songs = relationship("Song", secondary=song_tag_association, back_populates="tags")


class Song(Base):
    __tablename__ = "songs_master"

    song_id = Column(String, primary_key=True)



    name = Column(String)
    artist = Column(String)
    album_name = Column(String)
    genre = Column(String)
    year = Column(Integer)
    popularity = Column(Integer)
    explicit = Column(String)
    spotify_preview_url = Column(String)
    spotify_url = Column(String)
    album_images = Column(String)
    spotify_id = Column(String)
    duration_ms = Column(Integer)
    danceability = Column(String)
    energy = Column(String)
    key = Column(String)
    loudness = Column(String)
    mode = Column(String)
    speechiness = Column(String)
    acousticness = Column(String)
    instrumentalness = Column(String)
    liveness = Column(String)
    valence = Column(String)
    tempo = Column(String)
    time_signature = Column(String)
    n_tempo = Column(String)
    n_loudness = Column(String)

    tags = relationship("Tag", secondary=song_tag_association, back_populates="songs")