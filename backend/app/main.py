# app/main.py
import base64
import os
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
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

# Konfiguracja CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
    redirect_uri = os.getenv("SPOTIPY_REDIRECT_URI")

    # Walidacja konfiguracji
    if not client_id or not client_secret or not redirect_uri:
        raise ValueError(
            "Missing Spotify configuration. Please check your .env file:\n"
            f"SPOTIPY_CLIENT_ID: {'✓' if client_id else '✗'}\n"
            f"SPOTIPY_CLIENT_SECRET: {'✓' if client_secret else '✗'}\n"
            f"SPOTIPY_REDIRECT_URI: {'✓' if redirect_uri else '✗'}"
        )

    return SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=SPOTIFY_SCOPE,
        cache_path=None  # Wyłącz cache - używamy własnego zarządzania tokenami
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
def read_songs_by_tag(
        tag_name: str,
        limit: int = Query(default=10, ge=1),
        db: Session = Depends(get_db)
        ):
    # Bazowe zapytanie
    query = db.query(models.Song).filter(
        models.Song.tags_list.ilike(f"%{tag_name}%")
    )

    songs = query.limit(limit).all()
    if not songs:

        detail_msg = f"Nie znaleziono piosenek z tagiem '{tag_name}'"
        raise HTTPException(status_code=404, detail=detail_msg)

    return songs



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


##-------------------------SPOTIFY CONFIG-------------------------

@app.get("/spotify/config/check")
def check_spotify_config():
    """
    Sprawdza czy konfiguracja Spotify jest poprawna.
    """
    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
    redirect_uri = os.getenv("SPOTIPY_REDIRECT_URI")

    return {
        "configured": bool(client_id and client_secret and redirect_uri),
        "client_id": bool(client_id),
        "client_secret": bool(client_secret),
        "redirect_uri": bool(redirect_uri),
        "redirect_uri_value": redirect_uri if redirect_uri else None
    }


##-------------------------SPOTI-------------------------

@app.get("/login")
def login_spotify():
    """
    Krok 1: Przekierowuje użytkownika do logowania w Spotify.
    """
    sp_oauth = get_spotify_oauth()
    # Pobierz URL autoryzacji
    auth_url = sp_oauth.get_authorize_url()
    # Dodaj parametr show_dialog=true aby wymusić wyświetlenie ekranu logowania
    # nawet jeśli użytkownik jest już zalogowany w przeglądarce
    if '?' in auth_url:
        auth_url += '&show_dialog=true'
    else:
        auth_url += '?show_dialog=true'
    return RedirectResponse(auth_url)


@app.get("/callback")
def callback_spotify(code: str):
    """
    Krok 2: Spotify wraca tutaj z kodem. Wymieniamy go na token.
    """
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")

    try:
        sp_oauth = get_spotify_oauth()
        token_info = sp_oauth.get_access_token(code, as_dict=True, check_cache=False)

        if not token_info:
            print("ERROR: Failed to get token from Spotify")
            return RedirectResponse(f"{frontend_url}/?spotify_auth=error&reason=token_failed")

        # Zapisujemy token
        user_tokens['current_user'] = token_info
        print(f"SUCCESS: Token saved for user")

        # Przekieruj z powrotem do frontendu z komunikatem o sukcesie
        return RedirectResponse(f"{frontend_url}/?spotify_auth=success")

    except ValueError as e:
        # Błąd konfiguracji
        print(f"Configuration error: {e}")
        return RedirectResponse(f"{frontend_url}/?spotify_auth=error&reason=config")

    except Exception as e:
        error_message = str(e)
        print(f"ERROR in callback: {error_message}")

        # Sprawdź czy to błąd invalid_client
        if "invalid_client" in error_message.lower():
            print("ERROR: Invalid client credentials. Check SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET in .env")
            return RedirectResponse(f"{frontend_url}/?spotify_auth=error&reason=invalid_client")

        # Ogólny błąd
        return RedirectResponse(f"{frontend_url}/?spotify_auth=error&reason=unknown")


@app.get("/auth/status")
def check_auth_status():
    """
    Sprawdza czy użytkownik jest zalogowany do Spotify.
    """
    token_info = user_tokens.get('current_user')
    if not token_info:
        return {"authenticated": False}

    # Sprawdź czy token jest ważny
    sp_oauth = get_spotify_oauth()
    if sp_oauth.is_token_expired(token_info):
        # Spróbuj odświeżyć token
        try:
            token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
            user_tokens['current_user'] = token_info
        except Exception as e:
            print(f"Error refreshing token: {e}")
            user_tokens.pop('current_user', None)
            return {"authenticated": False}

    # Pobierz informacje o użytkowniku
    try:
        sp = spotipy.Spotify(auth=token_info['access_token'])
        user_info = sp.current_user()
        return {
            "authenticated": True,
            "user": {
                "id": user_info.get('id'),
                "display_name": user_info.get('display_name'),
                "email": user_info.get('email')
            }
        }
    except Exception as e:
        error_message = str(e)
        print(f"Error getting user info: {error_message}")

        # Jeśli błąd 403, to prawdopodobnie użytkownik nie jest dodany w Spotify Dashboard
        if "403" in error_message or "not be registered" in error_message:
            user_tokens.pop('current_user', None)
            return {
                "authenticated": False,
                "error": "spotify_user_not_registered",
                "message": "Musisz dodać swojego użytkownika do listy w Spotify Dashboard (Settings → Users and Access)"
            }

        user_tokens.pop('current_user', None)
        return {"authenticated": False}


@app.post("/auth/logout")
def logout_spotify():
    """
    Wylogowuje użytkownika (usuwa token z pamięci).
    """
    user_tokens.pop('current_user', None)
    return {"message": "Wylogowano pomyślnie"}







#Tutaj musi być wsadzane id
MOJA_LISTA_DO_PLAYLISTY = [
    "5cqaG09jwHAyDURuZXViwC",
    "4dDoIid58lgImNuYAxTRyM"
]

PLAYLIST_NAME = "Moja Playlista z Configu"       # Nazwa
PLAYLIST_DESC = "Opis ustawiony w zmiennej globalnej Python" # Opi s
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