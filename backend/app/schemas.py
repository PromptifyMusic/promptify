# app/schemas.py
from pydantic import BaseModel
from typing import Optional, Any, List


class SongBase(BaseModel):
    track_id: str
    name: Optional[str] = None
    artist: Optional[str] = None
    spotify_preview_url: Optional[str] = None
    tags: Optional[str] = None
    genre: Optional[str] = None
    year: Optional[int] = None
    duration_ms: Optional[int] = None
    danceability: Optional[float] = None
    energy: Optional[float] = None
    key: Optional[int] = None
    loudness: Optional[float] = None
    mode: Optional[int] = None
    speechiness: Optional[float] = None
    acousticness: Optional[float] = None
    instrumentalness: Optional[float] = None
    liveness: Optional[float] = None
    valence: Optional[float] = None
    tempo: Optional[float] = None
    time_signature: Optional[int] = None
    album_name: Optional[str] = None
    popularity: Optional[float] = None
    spotify_url: Optional[str] = None
    explicit: Optional[bool] = None
    album_images: Optional[Any] = None
    spotify_id: Optional[str] = None
    n_tempo: Optional[float] = None
    n_loudness: Optional[float] = None
    tags_list: Optional[str] = None
    tags_count: Optional[int] = None

    class Config:
        orm_mode = True  # pozwala Pydantic czytać z ORM obiektów


# app/schemas.py - dodaj na końcu

class PlaylistCreateRequest(BaseModel):
    name: str
    description: Optional[str] = "Playlista stworzona przez Songs API"
    song_ids: List[str]  # To będą track_id z Twojej bazy danych


class TempPlaylistRequest(BaseModel):
    quantity: int


class TempPlaylistTrack(BaseModel):
    track_id: str           # ID z naszej bazy (Million Song Dataset)
    spotify_id: str         # ID Spotify (potrzebne do API)
    name: str
    artist: str
    duration_ms: int
    image: Optional[str] = None

    class Config:
        orm_mode = True


class ExportPlaylistRequest(BaseModel):
    name: str
    description: Optional[str] = "Playlista wygenerowana przez Promptify"
    track_ids: List[str]  # Lista track_id ze Spotify (spotify_id z bazy)
    public: Optional[bool] = False


