# app/schemas.py
from pydantic import BaseModel
from typing import Optional, Any, List


class TagBase(BaseModel):
    tag_id: int
    name: str

    class Config:
        orm_mode = True


class SongMasterBase(BaseModel):
    song_id: str
    unnamed_0: Optional[str] = None
    name: Optional[str] = None
    artist: Optional[str] = None
    spotify_preview_url: Optional[str] = None
    genre: Optional[str] = None
    year: Optional[int] = None
    duration_ms: Optional[int] = None
    danceability: Optional[str] = None
    energy: Optional[str] = None
    key: Optional[str] = None
    loudness: Optional[str] = None
    mode: Optional[str] = None
    speechiness: Optional[str] = None
    acousticness: Optional[str] = None
    instrumentalness: Optional[str] = None
    liveness: Optional[str] = None
    valence: Optional[str] = None
    tempo: Optional[str] = None
    time_signature: Optional[str] = None
    album_name: Optional[str] = None
    popularity: Optional[int] = None
    spotify_url: Optional[str] = None
    explicit: Optional[str] = None
    album_images: Optional[str] = None
    spotify_id: Optional[str] = None
    n_tempo: Optional[str] = None
    n_loudness: Optional[str] = None

    tags: List[TagBase] = []

    class Config:
        orm_mode = True


class PlaylistCreateRequest(BaseModel):
    name: str
    description: Optional[str] = "Playlista stworzona przez Songs API"
    song_ids: List[str]




    #dla estetyki zmiuenic potem schemat na mniejszy