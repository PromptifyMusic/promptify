# app/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, Any, List


class SearchRequest(BaseModel):
    text: str = Field(..., description="Prompt użytkownika (np. 'energiczna muzyka do biegania')")
    top_n: int = Field(15, ge=1, le=50, description="Liczba utworów do zwrócenia")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "szybki rock",
                "top_n": 15
            }
        }


class PlaylistCreateRequest(BaseModel):
    name: str
    description: Optional[str] = "Playlista stworzona przez Songs API"
    song_ids: List[str]



class SongResult(BaseModel):
    name: str
    artist: Optional[str] = None
    spotify_id: Optional[str] = None
    album_images: Optional[str] = None
    duration_ms: Optional[int] = None

    class Config:
        from_attributes = True


class ReplaceSongRequest(BaseModel):
    text: str = Field(..., description="Oryginalny prompt (np. 'do biegania')")
    rejected_song_id: str = Field(..., description="ID piosenki do usunięcia")
    current_playlist_ids: List[str] = Field(..., description="Lista ID piosenek obecnych na playliście")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "szybki rock",
                "rejected_song_id": "3ZFTkvIE7kySmXDHtISCk8",
                "current_playlist_ids": ["3ZFTkvIE7kySmXDHtISCk8", "08mG3Y1vljYA6bvDtLLk68"]
            }
        }