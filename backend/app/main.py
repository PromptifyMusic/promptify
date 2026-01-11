# app/main.py
import base64
import os
from typing import List
from fastapi import FastAPI, Depends, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from . import models, schemas
from .database import SessionLocal, engine
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
from sqlalchemy import text
from . import engine
import numpy as np

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


# ENDPOINT API

def get_db():
    '''
    desc: Tworzy tymczasowe połączenie z bazą danych
    input: -
    output: Sesję bazy danych
    '''
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def startup_event():
    '''
    desc: uruchamia się przy starcie serwera, pobiera wektory tagów z bazy a następnie ładuje je do pamięci RAM
    input: -
    output: -
    '''
    print("[STARTUP]Gotowy.")



@app.post("/search/replace", response_model=schemas.SongResult)
def replace_song_endpoint(request: schemas.ReplaceSongRequest, db: Session = Depends(get_db) ):
    '''
        desc: Używa SI z engine by znaleźć zastępstwo dla wybranej piosenki z search
        input: Prompt usera, listę piosenek wyszukanych przez prompt oraz id piosenki jakiej już nie chcemy
        output: dane do nowej piosenki
    '''

    prompt = request.text
    exclude_ids = set(request.current_playlist_ids)
    if request.rejected_song_id:
        exclude_ids.add(request.rejected_song_id)

    extracted_phrases = engine.extract_relevant_phrases(prompt)

    classified_data = engine.classify_phrases_with_gliner(
        prompt,
        extracted_phrases,
        model=engine.model_gliner
    )

    queries = engine.prepare_queries_for_e5_separated(classified_data, prompt)
    tags_queries = queries['TAGS']
    audio_queries = queries['AUDIO']

    ROUTING_THRESHOLD = 0.81

    for phrase in list(audio_queries):
        check = engine.map_phrases_to_tags([phrase],db_session=db, threshold=ROUTING_THRESHOLD)
        if check:
            found_tag_name = list(check.keys())[0]
            audio_queries.remove(phrase)
            tags_queries.append(phrase)

    print(f"[REPLACE] Tagi={tags_queries} | Audio={audio_queries}")


    found_tags_map = engine.map_phrases_to_tags(tags_queries,db_session=db)
    query_tag_weights = engine.get_query_tag_weights(found_tags_map)


    criteria_audio = engine.phrases_to_features(
        audio_queries,
        search_indices=engine.SEARCH_INDICES,
        lang_code="pl"
    )


    candidates_df = engine.fetch_candidates_from_db(query_tag_weights, db,criteria_audio, limit=200)

    if candidates_df.empty:
        print("[REPLACE] Brak kandydatów po tagach. Fallback popularne.")
        candidates_df = engine.fetch_candidates_from_db({}, db, criteria_audio, limit=200)

    criteria_audio = engine.phrases_to_features(
        audio_queries,
        search_indices=engine.SEARCH_INDICES,
        lang_code="pl"
    )
    audio_scores = engine.calculate_audio_match(candidates_df, criteria_audio)
    candidates_df['audio_score'] = audio_scores

    has_tags = bool(found_tags_map)
    merged_df = engine.merge_tag_and_audio_scores(candidates_df, audio_weight=0.6, use_tags=has_tags)

    sorted_candidates = merged_df.sort_values("score", ascending=False)

    replacement_song = None

    for index, row in sorted_candidates.iterrows():
        s_id = row['spotify_id']

        if s_id not in exclude_ids:
            replacement_song = row
            break
        else:
            pass

    if replacement_song is None:
        raise HTTPException(status_code=404, detail="Nie znaleziono unikalnego utworu do wymiany.")

    result_cols = [
        "spotify_id", "name", "artist", "popularity", "score",
        "spotify_preview_url", "album_images", "duration_ms"
    ]

    available_cols = [c for c in result_cols if c in replacement_song.index]

    return replacement_song[available_cols].replace({np.nan: None}).to_dict()





@app.post("/search", response_model=List[schemas.SongResult])
def search_songs(
        request: schemas.SearchRequest,
        # text: str = Query(..., description="Prompt użytkownika"),
        # ilosc: int = Query(15, alias="top_n", ge=1, le=50),
        # Wstrzykujemy sesję bazy danych (KLUCZOWE dla nowej wersji)
        db: Session = Depends(get_db)):
    '''
        desc: analizuje tekst przy pomocy SI a następnie szuka pasujących piosenek na podstawie tagów i cech audio
        input: prompt, liczba utworów
        output: liste piosenek pasujących do promptu o liczbie jaką podaliśmy
    '''

    text = request.text
    top_n = request.top_n


    print(f"\n[API] NOWY PROMPT: '{text}' (Top {top_n})")

    extracted_phrases = engine.extract_relevant_phrases(text)

    classified_data = engine.classify_phrases_with_gliner(
        text,
        extracted_phrases,
        model=engine.model_gliner
    )

    queries = engine.prepare_queries_for_e5_separated(classified_data, text)
    tags_queries = queries['TAGS']
    audio_queries = queries['AUDIO']





    #Sito
    for phrase in list(audio_queries):
        check = engine.map_phrases_to_tags([phrase],db_session=db, threshold=0.81)

        if check:
            found_tag_name = list(check.keys())[0]
            audio_queries.remove(phrase)
            tags_queries.append(phrase)



    print(f"Tagi={tags_queries} | Audio={audio_queries}")

    found_tags_map = engine.map_phrases_to_tags(tags_queries, db_session=db)
    query_tag_weights = engine.get_query_tag_weights(found_tags_map)

    criteria_audio = engine.phrases_to_features(audio_queries, search_indices=engine.SEARCH_INDICES, lang_code="pl")

    candidates_df = engine.fetch_candidates_from_db(query_tag_weights, db, criteria_audio)

    if candidates_df.empty:
        print("[API] Brak wyników po tagach. Pobieranie puli zapasowej.")
        candidates_df = engine.fetch_candidates_from_db({}, db, criteria_audio,  limit=200)

    audio_scores = engine.calculate_audio_match(candidates_df,  criteria_audio)
    candidates_df['audio_score'] = audio_scores

    has_tags = bool(found_tags_map)
    merged_df = engine.merge_tag_and_audio_scores(candidates_df, audio_weight=0.6, use_tags=has_tags)

    t_high, t_mid = engine.calculate_dynamic_thresholds(merged_df)
    tier_a, tier_b, tier_c = engine.tier_by_score(merged_df, t_high, t_mid)

    working_set = engine.build_working_set(
        tier_a, tier_b, tier_c,
        target_pool_size=engine.WORKSET_CONFIG['target_pool_size'],
        min_required_size=engine.WORKSET_CONFIG['min_required_size'],
        popularity_rescue_ratio=engine.WORKSET_CONFIG['popularity_rescue_ratio']
    )

    sampling_cfg = engine.SAMPLING_CONFIG.copy()
    sampling_cfg['final_n'] = top_n

    final_playlist = engine.sample_final_songs(
        working_set,
        popularity_cfg=engine.POPULARITY_CONFIG,
        sampling_cfg=sampling_cfg
    )

    if final_playlist.empty:
        raise HTTPException(status_code=404, detail="Brak utworów.")

    result_cols = ["spotify_id", "name", "artist", "popularity", "score", "spotify_preview_url", "album_images",
                   "duration_ms"]
    available_cols = [c for c in result_cols if c in final_playlist.columns]

    return final_playlist[available_cols].replace({np.nan: None}).to_dict(orient="records")


#Mikolaj

# KONFIGURACJA SPOTIFY
# Zakres uprawnień (Scope). Musimy poprosić o prawo do edycji playlist.
SPOTIFY_SCOPE = "playlist-modify-public playlist-modify-private"




@app.get("/")
def root():
    '''
        desc: prosty endpoint do kontroli czy api działa
        input: -
        output: -
    '''
    return {"message": "Api  działa"}




##-------------------------SPOTIFY CONFIG-------------------------

def get_spotify_oauth():
    '''
            desc: Konfiguruje mechanizm logowania
            input: -
            output: Ustawienia do bezpiecznego połączenia
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
        cache_path=None
    )

user_tokens = {}


@app.get("/spotify/config/check")
def check_spotify_config():
    '''
        desc: Sprawdza czy w pliku .env są wszystkie potrzebne klucze do Spotify
        input: -
        output: Raport
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
    '''
        desc: Rozpoczyna logowanie użytkownika
        input: -
        output: Przekierowanie na oficjalną stronę Spotify
    '''
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
    '''
        desc: Wymienia kod na token
        input: Specjalny kod autoryzacyjny zwracany przez spotify
        output: Przekierowanie z powrotem na naszą stronę
    '''
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")


    if error:
        print(f"INFO: User cancelled authorization. Error: {error}")
        return RedirectResponse(f"{frontend_url}/?spotify_auth=cancelled")

    if not code:
        print("ERROR: No code or error parameter received")
        return RedirectResponse(f"{frontend_url}/?spotify_auth=error&reason=no_code")

    try:
        sp_oauth = get_spotify_oauth()
        token_info = sp_oauth.get_access_token(code, as_dict=True, check_cache=False)

        if not token_info:
            print("ERROR: Failed to get token from Spotify")
            return RedirectResponse(f"{frontend_url}/?spotify_auth=error&reason=token_failed")

        user_tokens['current_user'] = token_info
        print(f"SUCCESS: Token saved for user")

        return RedirectResponse(f"{frontend_url}/?spotify_auth=success")

    except ValueError as e:
        print(f"Configuration error: {e}")
        return RedirectResponse(f"{frontend_url}/?spotify_auth=error&reason=config")

    except Exception as e:
        error_message = str(e)
        print(f"ERROR in callback: {error_message}")

        if "invalid_client" in error_message.lower():
            print("ERROR: Invalid client credentials. Check SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET in .env")
            return RedirectResponse(f"{frontend_url}/?spotify_auth=error&reason=invalid_client")

        return RedirectResponse(f"{frontend_url}/?spotify_auth=error&reason=unknown")


@app.get("/auth/status")
def check_auth_status():
    '''
        desc: Sprawdza, czy użytkownik jest zalogowany i czy ma aktywną sesję
        input: -
        output: Dane zalogowanego użytkownika
    '''
    token_info = user_tokens.get('current_user')
    if not token_info:
        return {"authenticated": False}

    sp_oauth = get_spotify_oauth()
    if sp_oauth.is_token_expired(token_info):
        try:
            token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
            user_tokens['current_user'] = token_info
        except Exception as e:
            print(f"Error refreshing token: {e}")
            user_tokens.pop('current_user', None)
            return {"authenticated": False}

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
    '''
        desc: Wylogowuje użytkownika
        input: -
        output: Komunikat
    '''
    user_tokens.pop('current_user', None)
    return {"message": "Wylogowano pomyślnie"}




#testowa itp
MOJA_LISTA_DO_PLAYLISTY = [
    "5cqaG09jwHAyDURuZXViwC",
    "4dDoIid58lgImNuYAxTRyM"
]

PLAYLIST_NAME = "Moja Playlista z Configu"
PLAYLIST_DESC = "Opis ustawiony w zmiennej globalnej Python"
PLAYLIST_PUBLIC = False

PLAYLIST_COVER_PATH = "cover.jpg"

#do zdjęc
def encode_image_to_base64(image_path: str):
    '''
        desc: Wylogowuje użytkownika
        input: Zamienia obrazek na ciąg znaków tekstowych
        output: Zakodowane zdjęcie
    '''
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        return None



@app.post("/create_playlist_hardcoded")
def create_playlist_hardcoded():
    '''
        desc: Funkcja testowa, która tworzy playlistę (do delete?)
        input: -
        output: Status playlisty
    '''

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



@app.post("/export_playlist")
def export_playlist(request: schemas.PlaylistCreateRequest):
    """
    Eksportuje playlistę do Spotify na podstawie danych z requestu.

    Input:
        - request.name (str): Nazwa playlisty
        - request.description (str, optional): Opis playlisty
        - request.song_ids (List[str]): Lista Spotify ID utworów
        - request.public (bool, optional): Czy playlista ma być publiczna (domyślnie False)

    Output:
        - status (str): Status operacji
        - message (str): Komunikat
        - playlist_id (str): ID utworzonej playlisty
        - playlist_url (str): URL do playlisty
        - playlist_name (str): Nazwa playlisty
        - tracks_count (int): Liczba dodanych utworów
        - public (bool): Czy playlista jest publiczna
    """

    # Autoryzacja
    token_info = user_tokens.get('current_user')
    if not token_info:
        raise HTTPException(status_code=401, detail="Najpierw zaloguj się na /login")

    # Sprawdź czy token nie wygasł
    sp_oauth = get_spotify_oauth()
    if sp_oauth.is_token_expired(token_info):
        try:
            token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
            user_tokens['current_user'] = token_info
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Token wygasł i nie można go odświeżyć: {str(e)}")

    try:
        sp = spotipy.Spotify(auth=token_info['access_token'])
        user_id = sp.current_user()['id']

        # Konwersja ID na Spotify URI
        spotify_uris = []
        for sid in request.song_ids:
            if "spotify:track:" not in sid:
                spotify_uris.append(f"spotify:track:{sid}")
            else:
                spotify_uris.append(sid)

        # Tworzenie playlisty
        playlist = sp.user_playlist_create(
            user=user_id,
            name=request.name,
            public=request.public,
            description=request.description
        )

        # Dodawanie utworów do playlisty
        if spotify_uris:
            # Spotify API akceptuje max 100 utworów na raz
            for i in range(0, len(spotify_uris), 100):
                batch = spotify_uris[i:i + 100]
                sp.playlist_add_items(playlist_id=playlist['id'], items=batch)

        return {
            "status": "success",
            "message": "Playlista została pomyślnie utworzona w Spotify",
            "playlist_id": playlist['id'],
            "playlist_url": playlist['external_urls']['spotify'],
            "playlist_name": request.name,
            "tracks_count": len(spotify_uris),
            "public": request.public
        }

    except spotipy.exceptions.SpotifyException as e:
        error_msg = str(e)
        if "403" in error_msg or "Forbidden" in error_msg:
            raise HTTPException(
                status_code=403,
                detail="Brak uprawnień. Upewnij się, że zalogowany użytkownik ma odpowiednie uprawnienia w Spotify."
            )
        raise HTTPException(status_code=500, detail=f"Błąd Spotify API: {error_msg}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd podczas tworzenia playlisty: {str(e)}")
