# app/main.py
import base64
import os
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, Depends, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from . import models, schemas
SongModel = models.Song
TagModel = models.Tag
from .database import SessionLocal, engine
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from gliner import GLiNER
import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect, DetectorFactory
from sqlalchemy import text
from . import engine


load_dotenv()

app = FastAPI(title="Songs API")

#Konfiguracja CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# ENDPOINT API
# ==========================================
# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()




#usunac genre kolumna


@app.post("/search/replace", response_model=schemas.SongResult)
def replace_song_endpoint(
        request: schemas.ReplaceSongRequest,
        db: Session = Depends(get_db)
):
    """
    input: request (ReplaceSongRequest) - JSON z promptem, odrzuconym ID i listą obecnych ID
    output: dict - pojedyncza  piosenka (najlepszy dostępny zastępca)
    """

    prompt = request.text
    # Budujemy zbiór ID, których nie chcemy
    exclude_ids = set(request.current_playlist_ids)
    exclude_ids.add(request.rejected_song_id)

    print(f"\n[REPLACE] Start wymiany dla: '{request.rejected_song_id}'", flush=True)

    # --- 1. RE-USE LOGIKI WYSZUKIWANIA ---

    # NLP
    extracted_phrases = engine.extract_relevant_phrases(prompt)
    tags_queries = extracted_phrases
    audio_queries = extracted_phrases

    # SQL Search (Tagi)
    found_tags_map = engine.search_tags_in_db(tags_queries, db, engine.model_e5,
                                       threshold=0.45)  # Lekko niższy próg dla bezpieczeństwa
    query_tag_weights = engine.get_query_tag_weights(found_tags_map)

    # DB Fetch
    candidates_df = engine.fetch_candidates_from_db(query_tag_weights, db, limit=50)

    # Fallback (gdyby nic nie znalazł po tagach)
    if candidates_df.empty:
        print("[REPLACE]", flush=True)
        candidates_df = engine.fetch_candidates_from_db({}, db, limit=50)

    # Audio Match
    criteria_audio = engine.phrases_to_features(audio_queries,engine.SEARCH_INDICES ,lang_code="pl")

    audio_scores = engine.calculate_audio_match(candidates_df, criteria_audio)
    candidates_df['audio_score'] = audio_scores

    # Merge & Final Score
    has_tags = bool(found_tags_map)
    # [POPRAWKA] Wpisano wagę na sztywno (0.6), zamiast szukać w configu
    merged_df = engine.merge_tag_and_audio_scores(candidates_df, audio_weight=0.6, use_tags=has_tags)

    # --- 2. FILTROWANIE (Znalezienie zastępcy) ---

    # Sortujemy od najlepszego dopasowania
    sorted_candidates = merged_df.sort_values("score", ascending=False)

    replacement_song = None

    for index, row in sorted_candidates.iterrows():
        s_id = row['spotify_id']

        # Sprawdzamy czy to ID jest na czarnej liście
        if s_id not in exclude_ids:
            replacement_song = row
            print(f"[REPLACE] Znaleziono zastępstwo: '{row['name']}' (Score: {row['score']:.4f})", flush=True)
            break

    if replacement_song is None:
        print("[REPLACE] Nie znaleziono ", flush=True)
        raise HTTPException(status_code=404, detail="Nie znaleziono więcej pasujących utworów do wymiany.")

    # --- 3. PRZYGOTOWANIE WYNIKU ---

    # [POPRAWKA] Usunięto 'spotify_preview_url', żeby nie wywalało błędu
    result_cols = [
        "spotify_id", "name", "artist", "popularity", "score",
        "album_images", "duration_ms"
    ]

    # Konwersja (replace NaN na None dla poprawnego JSONa)
    result_dict = replacement_song[result_cols].replace({np.nan: None}).to_dict()

    return result_dict


@app.post("/search", response_model=List[schemas.SongResult])
def search_songs(
        request: schemas.SearchRequest,
        # Wstrzykujemy sesję bazy danych (KLUCZOWE dla nowej wersji)
        db: Session = Depends(get_db)):

    """
        Wejście:
            - request.text (str): Tekst zapytania użytkownika (prompt, np. "szybki rock do biegania").
            - request.top_n (int): Oczekiwana długość playlisty (domyślnie 15).
            - db (Session): Aktywna sesja połączenia z bazą danych.

        Wyjście:
            - List[dict]: Lista słowników JSON, gdzie każdy element to sformatowany utwór zawierający m.in. id, nazwę, artystę, okładkę i wynik dopasowania (score).

        Opis:
            Główny orkiestrator silnika rekomendacji. Realizuje pełny pipeline przetwarzania:
            1. Ekstrakcja fraz kluczowych z tekstu.
            2. Znalezienie pasujących tagów w bazie wektorowej.
            3. Pobranie wstępnej listy kandydatów z bazy SQL.
            4. Obliczenie dopasowania audio i połączenie go z wynikiem tagów (hybrydowa punktacja).
            5. Podział wyników na poziomy jakości  i finalne, ważone losowanie utworów z uwzględnieniem ich popularności.
      """
    # Przypisanie zmiennych z parametrów
    prompt = request.text
    final_n = request.top_n

    print(f"\nNOWE ZAPYTANIE: '{prompt}' (Top {final_n})")


    # 1. NLP & EMBEDDINGS
    extracted_phrases = engine.extract_relevant_phrases(prompt)
    tags_queries = extracted_phrases
    audio_queries = extracted_phrases

    # 2. SZUKANIE TAGÓW (SQL pgvector)
    found_tags_map = engine.search_tags_in_db(tags_queries, db, engine.model_e5, threshold=0.65)
    query_tag_weights = engine.get_query_tag_weights(found_tags_map)

    # 3. POBIERANIE KANDYDATÓW (SQL WHERE)
    candidates_df = engine.fetch_candidates_from_db(query_tag_weights, db, limit=engine.RETRIEVAL_CONFIG["n_candidates"])

    # Fallback
    if candidates_df.empty:
        print("Brak wyników po tagach. Pobieranie losowych popularnych.")
        candidates_df = engine.fetch_candidates_from_db({}, db, limit=100)

    # 4. AUDIO MATCH
    criteria_audio = engine.phrases_to_features(audio_queries, engine.SEARCH_INDICES, lang_code="pl")
    audio_scores = engine.calculate_audio_match(candidates_df, criteria_audio)
    candidates_df['audio_score'] = audio_scores

    # 5. MERGE
    has_tags = bool(found_tags_map)
    merged_df = engine.merge_tag_and_audio_scores(candidates_df, audio_weight=engine.SCORING_CONFIG['audio_weight'],
                                           use_tags=has_tags)

    # 6. TIEROWANIE
    t_high, t_mid = engine.calculate_dynamic_thresholds(
        merged_df,
        high_threshold=engine.WORKSET_CONFIG['min_absolute_high'],
        mid_threshold=engine.WORKSET_CONFIG['min_absolute_mid']
    )
    tier_a, tier_b, tier_c = engine.tier_by_score(merged_df, t_high, t_mid)

    # 7. PUL ROBOCZA
    working_set = engine.build_working_set(
        tier_a, tier_b, tier_c,
        target_pool_size=engine.WORKSET_CONFIG['target_pool_size'],
        min_required_size=engine.WORKSET_CONFIG['min_required_size'],
        popularity_rescue_ratio=engine.WORKSET_CONFIG['popularity_rescue_ratio']
    )

    # 8. FINALNE LOSOWANIE
    final_playlist = engine.sample_final_songs(
        working_set,
        popularity_cfg=engine.POPULARITY_CONFIG,
        sampling_cfg={
            "final_n": final_n,
            "alpha": 2.0,
            "shuffle": True
        }
    )

    # 9. ZWROT WYNIKÓW
    if final_playlist.empty:
        raise HTTPException(status_code=404, detail="Nie udało się znaleźć pasujących utworów.")

    result_cols = [
        "spotify_id", "name", "artist", "popularity", "score",
        "spotify_preview_url", "album_images", "duration_ms"
    ]
    available_cols = [c for c in result_cols if c in final_playlist.columns]

    return final_playlist[available_cols].to_dict(orient="records")








#Mikolaj

# KONFIGURACJA SPOTIFY
# Zakres uprawnień (Scope). Musimy poprosić o prawo do edycji playlist.
SPOTIFY_SCOPE = "playlist-modify-public playlist-modify-private"




@app.get("/")
def root():
    return {"message": "Api  działa"}




##-------------------------SPOTIFY CONFIG-------------------------

def get_spotify_oauth():
    '''
    input: None (korzysta ze zmiennych środowiskowych: CLIENT_ID, CLIENT_SECRET, REDIRECT_URI)
    output: spotipy.oauth2.SpotifyOAuth - skonfigurowany obiekt menedżera autoryzacji
    description: Inicjalizuje mechanizm OAuth2 dla Spotify. Pobiera klucze API z pliku .env i przeprowadza ich walidację, rzucając błąd w przypadku braku konfiguracji.
                 Parametr `cache_path=None` celowo wyłącza domyślne zapisywanie tokenu w pliku `.cache`,
                 ponieważ w architekturze wieloużytkownikowej tokeny są zarządzane dynamicznie w pamięci (słownik `user_tokens`) lub w bazie danych.
    '''

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


@app.get("/spotify/config/check")
def check_spotify_config():
    '''
    Wejście: Brak (korzysta ze zmiennych środowiskowych .env: CLIENT_ID, SECRET, REDIRECT_URI).
    Wyjście: Obiekt klasy spotipy.oauth2.SpotifyOAuth.
    Opis: Inicjalizuje menedżera autoryzacji. Sprawdza obecność kluczy w pliku .env i konfiguruje klienta OAuth bez zapisywania tokenów w pliku cache (cache_path=None).
    '''
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
def callback_spotify(code: str = None, error: str = None):
    """
    Krok 2: Spotify wraca tutaj z kodem lub błędem.
    - Sukces: ?code=XXX
    - Anulowanie: ?error=access_denied
    """
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")

    # Użytkownik anulował autoryzację
    if error:
        print(f"INFO: User cancelled authorization. Error: {error}")
        return RedirectResponse(f"{frontend_url}/?spotify_auth=cancelled")

    # Brak kodu autoryzacyjnego
    if not code:
        print("ERROR: No code or error parameter received")
        return RedirectResponse(f"{frontend_url}/?spotify_auth=error&reason=no_code")

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




#testowa itp
MOJA_LISTA_DO_PLAYLISTY = [
    "5cqaG09jwHAyDURuZXViwC",
    "4dDoIid58lgImNuYAxTRyM"
]

PLAYLIST_NAME = "Moja Playlista z Configu"       # Nazwa
PLAYLIST_DESC = "Opis ustawiony w zmiennej globalnej Python" # Opi s
PLAYLIST_PUBLIC = False                          # Czy publiczna? (True/False)

PLAYLIST_COVER_PATH = "cover.jpg"

#do zdjęc
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

    #Autoryzacja
    token_info = user_tokens.get('current_user')
    if not token_info:
        raise HTTPException(status_code=401, detail="Najpierw zaloguj się na /login")

    sp = spotipy.Spotify(auth=token_info['access_token'])
    user_id = sp.current_user()['id']

    #Pobranie danych ze zmiennych globalnych
    current_ids = MOJA_LISTA_DO_PLAYLISTY
    pl_name = PLAYLIST_NAME
    pl_public = PLAYLIST_PUBLIC
    pl_desc = PLAYLIST_DESC
    cover_path = PLAYLIST_COVER_PATH

    #Pętla przetwarzająca ID
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


    #Tworzenie playlisty
    playlist = sp.user_playlist_create(
        user=user_id,
        name=pl_name,
        public=pl_public,
        description=pl_desc
    )

    #Dodawanie zdjęcia (jeśli jest w configu)
    cover_msg = "Brak zdjęcia"
    if cover_path:
        img_base64 = encode_image_to_base64(cover_path)
        if img_base64:
            try:
                sp.playlist_upload_cover_image(playlist['id'], img_base64)
                cover_msg = "Zdjęcie dodane"
            except Exception as e:
                cover_msg = f"Błąd zdjęcia: {e}"

    #Wrzucanie utworów
    if spotify_uris:
        sp.playlist_add_items(playlist_id=playlist['id'], items=spotify_uris)

    return {
        "status": "Gotowe!",
        "playlist_url": playlist['external_urls']['spotify'],
        "cover_status": cover_msg,
        "tracks_count": len(spotify_uris)
    }