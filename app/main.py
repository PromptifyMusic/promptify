# app/main.py
import os
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from . import models, schemas
from .database import SessionLocal, engine
from typing import List, Optional
from sqlalchemy import or_

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

# Jeżeli chcesz tworzyć tabele z modeli (tylko gdy nie masz już tabeli)
# models.Base.metadata.create_all(bind=engine)
load_dotenv()

app = FastAPI(title="Songs API")

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- KONFIGURACJA SPOTIFY ---
# Zakres uprawnień (Scope). Musimy poprosić o prawo do edycji playlist.
SPOTIFY_SCOPE = "playlist-modify-public playlist-modify-private"

def get_spotify_oauth():
    return SpotifyOAuth(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
        scope=SPOTIFY_SCOPE
    )

user_tokens = {}


@app.get("/songs/all", response_model=list[schemas.SongBase])
def read_songs(limit: int = 1000, offset: int = 0, db: Session = Depends(get_db)):

    songs = db.query(models.Song).limit(limit).offset(offset).all()
    return songs

@app.get("/")
def root():
    return {"message": "Api działa"}



@app.get("/songs/{tag_name}", response_model=list[schemas.SongBase])
def read_songs_by_tag(tag_name: str, db: Session = Depends(get_db)):

    # Używamy ILIKE, żeby nie rozróżniało wielkości liter
    songs = db.query(models.Song).filter(models.Song.tags_list.ilike(f"%{tag_name}%")).all()

    if not songs:
        raise HTTPException(status_code=404, detail=f"Nie znaleziono piosenek z tagiem '{tag_name}'")

    return songs

##dodać zapiś do spoti





## //w parametrze ilosc, - tego na razie nie
## wiele argumentów.
## Doker z,
## podmienianie jednego utworu
## filtr po parametrach
## dodanie dockera
## dodanie rzeczy na gita


@app.get("/songs", response_model=list[schemas.SongBase])
def read_songs(
    q: str | None = None,   # <--- ogólny filtr
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    query = db.query(models.Song)

    if q:
        query = query.filter(
            models.Song.name.ilike(f"%{q}%") |
            models.Song.artist.ilike(f"%{q}%") |
            models.Song.tags_list.ilike(f"%{q}%")
        )

    songs = query.limit(limit).offset(offset).all()
    return songs





##-------------------------SPOTI-------------------------

@app.get("/login")
def login_spotify():
    """
    Krok 1: Przekierowuje użytkownika do logowania w Spotify.
    """
    sp_oauth = get_spotify_oauth()
    auth_url = sp_oauth.get_authorize_url()
    return RedirectResponse(auth_url)


@app.get("/callback")
def callback_spotify(code: str):
    """
    Krok 2: Spotify wraca tutaj z kodem. Wymieniamy go na token.
    """
    sp_oauth = get_spotify_oauth()
    token_info = sp_oauth.get_access_token(code)

    # Zapisujemy token (w uproszczonym modelu globalnym)
    # W prawdziwej aplikacji powinieneś użyć cookies lub zwrócić token do frontendu.
    user_tokens['current_user'] = token_info

    return {"message": "Zalogowano pomyślnie! Możesz teraz tworzyć playlisty."}


# app/main.py - dodaj na końcu

@app.get("/create_playlist_hardcoded")
def create_playlist_hardcoded():
    """
    Tworzy playlistę z listy piosenek zdefiniowanej "na sztywno" w kodzie.
    Nie łączy się z bazą danych SQL, operuje tylko na zmiennych.
    """

    # 1. AUTORYZACJA
    token_info = user_tokens.get('current_user')
    if not token_info:
        raise HTTPException(status_code=401, detail="Najpierw zaloguj się na /login")

    sp = spotipy.Spotify(auth=token_info['access_token'])
    user_id = sp.current_user()['id']

    # WAŻNE: Wklej tutaj prawdziwe ID ze swojej bazy/Spotify!
    my_hardcoded_songs = [
        {
            "name": "Piosenka Testowa 1",
            "spotify_id": "5cqaG09jwHAyDURuZXViwC",  # <--- WKLEJ TU ID
        },
        {
            "name": "Piosenka Testowa 2",
            "spotify_id": "4dDoIid58lgImNuYAxTRyM"  # <--- WKLEJ TU ID
        }
    ]

    # 3. PRZETWARZANIE (PĘTLA W PAMIĘCI)
    # Nie pytamy bazy SQL, po prostu mielimy listę, którą mamy wyżej.
    spotify_uris = []
    for song in my_hardcoded_songs:
        sid = song["spotify_id"]
        # Dodajemy prefix, jeśli go nie ma
        if "spotify:track:" not in sid:
            spotify_uris.append(f"spotify:track:{sid}")
        else:
            spotify_uris.append(sid)

    # 4. TWORZENIE PLAYLISTY
    if not spotify_uris:
        return {"error": "Lista utworów jest pusta lub ma błędne ID"}

    playlist = sp.user_playlist_create(
        user=user_id,
        name="Playlista Na Sztywno",
        public=False,
        description="Stworzona z ręcznie wpisanej listy w kodzie"
    )

    # 5. WRZUCANIE UTWORÓW
    sp.playlist_add_items(playlist_id=playlist['id'], items=spotify_uris)

    return {
        "status": "Gotowe!",
        "playlist_url": playlist['external_urls']['spotify'],
        "tracks_count": len(spotify_uris)
    }