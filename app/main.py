# app/main.py
import base64
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

# KONFIGURACJA SPOTIFY
# Zakres uprawnień (Scope). Musimy poprosić o prawo do edycji playlist.
SPOTIFY_SCOPE = "playlist-modify-public playlist-modify-private"

def get_spotify_oauth():
    return SpotifyOAuth(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SEno CRET"),
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

    # Zapisujemy token
    # dodać cookies (?)
    user_tokens['current_user'] = token_info

    return {"message": "Zalogowano pomyślnie! Możesz teraz tworzyć playlisty."}







#Tutaj musi być wsadzane id
MOJA_LISTA_DO_PLAYLISTY = [
    "5cqaG09jwHAyDURuZXViwC",
    "4dDoIid58lgImNuYAxTRyM"
]

PLAYLIST_NAME = "Moja Playlista z Configu"       # Nazwa
PLAYLIST_DESC = "Opis ustawiony w zmiennej globalnej Python" # Opis
PLAYLIST_PUBLIC = False                          # Czy publiczna? (True/False)

PLAYLIST_COVER_PATH = "cover.jpg"


# Funkcja pomocnicza (musi być w kodzie, żeby zdjęcie działało)
def encode_image_to_base64(image_path: str):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        return None


@app.post("/create_playlist_hardcoded")
def create_playlist_hardcoded():
    """
    Wersja z zewnętrznymi zmiennymi + okładką + POPRAWNYMI WCIĘCIAMI.
    """

    # A. Autoryzacja
    token_info = user_tokens.get('current_user')
    if not token_info:
        raise HTTPException(status_code=401, detail="Najpierw zaloguj się na /login")

    sp = spotipy.Spotify(auth=token_info['access_token'])
    user_id = sp.current_user()['id']

    # B. Pobranie danych ze zmiennych globalnych
    current_ids = MOJA_LISTA_DO_PLAYLISTY
    pl_name = PLAYLIST_NAME
    pl_public = PLAYLIST_PUBLIC
    pl_desc = PLAYLIST_DESC
    cover_path = PLAYLIST_COVER_PATH

    # C. Pętla przetwarzająca ID
    spotify_uris = []
    for sid in current_ids:
        if "spotify:track:" not in sid:
            spotify_uris.append(f"spotify:track:{sid}")
        else:
            spotify_uris.append(sid)

    # ---------------------------------------------------------
    # KLUCZOWY MOMENT: Tu kończy się pętla.
    # Kod poniżej jest przesunięty w lewo (nie ma wcięcia).
    # Wykona się TYLKO RAZ.
    # ---------------------------------------------------------

    # D. Tworzenie playlisty
    playlist = sp.user_playlist_create(
        user=user_id,
        name=pl_name,
        public=pl_public,
        description=pl_desc
    )

    # E. Dodawanie zdjęcia (jeśli jest w configu)
    cover_msg = "Brak zdjęcia"
    if cover_path:
        img_base64 = encode_image_to_base64(cover_path)
        if img_base64:
            try:
                sp.playlist_upload_cover_image(playlist['id'], img_base64)
                cover_msg = "Zdjęcie dodane"
            except Exception as e:
                cover_msg = f"Błąd zdjęcia: {e}"

    # F. Wrzucanie utworów
    if spotify_uris:
        sp.playlist_add_items(playlist_id=playlist['id'], items=spotify_uris)

    return {
        "status": "Gotowe!",
        "playlist_url": playlist['external_urls']['spotify'],
        "cover_status": cover_msg,
        "tracks_count": len(spotify_uris)
    }