# Backend Promptify

API backendu dla aplikacji Promptify - generowanie playlist Spotify na podstawie tagów muzycznych.

##  Szybki Start

### Wymagania
- Python 3.10+
- PostgreSQL (uruchomiony serwer)

### Automatyczna instalacja (ZALECANE dla Windows)
Upewnij się, że znajdujesz się w katalogu `backend` i uruchamiasz skrypty za pomocą PowerShell:
```powershell
# 1. Przygotuj backend (jednorazowo)
.\setup_backend.ps1

# 2. Uruchom backend
.\run_backend.ps1
```

**Skrypt `setup_backend.ps1` automatycznie:**
-  Sprawdzi Python i PostgreSQL
-  Utworzy środowisko wirtualne
-  Zainstaluje wszystkie zależności
-  Skonfiguruje plik .env
-  Sprawdzi połączenie z bazą danych

**Skrypt `run_backend.ps1` automatycznie:**
-  Aktywuje środowisko wirtualne
-  Sprawdzi PostgreSQL
-  Uruchomi serwer FastAPI

### Instalacja ręczna (Windows)

```powershell
# 1. Przejdź do katalogu backend
cd C:\workspace\promptify\backend

# 2. Utwórz i aktywuj środowisko wirtualne
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Zainstaluj zależności
pip install -r requirements.txt
```

## Konfiguracja

### 1. Baza danych PostgreSQL

Utwórz bazę danych i tabelę:

```sql
-- Połącz się z PostgreSQL
psql -U postgres

-- Utwórz tabelę spotify_tracks
CREATE TABLE spotify_tracks (
    track_id text PRIMARY KEY,
    name text,
    artist text,
    spotify_preview_url text,
    tags text,
    genre text,
    year integer,
    duration_ms integer,
    danceability double precision,
    energy double precision,
    key integer,
    loudness double precision,
    mode integer,
    speechiness double precision,
    acousticness double precision,
    instrumentalness double precision,
    liveness double precision,
    valence double precision,
    tempo double precision,
    time_signature integer,
    album_name text,
    popularity double precision,
    spotify_url text,
    explicit boolean,
    album_images jsonb,
    spotify_id text,
    n_tempo double precision,
    n_loudness double precision,
    tags_list text,
    tags_count integer
);
```

### 2. Plik `.env`

**WAŻNE: Plik `.env` zawiera wrażliwe dane i NIE JEST commitowany do Git!**

Skopiuj `env_example.txt` do `.env` i uzupełnij swoimi danymi:


Następnie edytuj plik `.env` i uzupełnij:

```env
DATABASE_URL=postgresql://postgres:TWOJE_HASLO@localhost:5432/postgres

SPOTIPY_CLIENT_ID=twoje_client_id
SPOTIPY_CLIENT_SECRET=twoje_client_secret
SPOTIPY_REDIRECT_URI=http://127.0.0.1:8000/callback
```

**Spotify API Credentials:**
1. Przejdź do [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Utwórz aplikację
3. Skopiuj Client ID i Client Secret
4. Dodaj Redirect URI: `http://127.0.0.1:8000/callback`

### 3. Przykładowe dane (opcjonalnie)

Wstaw przykładowe utwory do bazy:

```powershell
python populate_database.py
```

## Uruchomienie

```powershell
# Aktywuj środowisko (jeśli nieaktywne)
.\venv\Scripts\Activate.ps1   # Windows

# Uruchom serwer
python -m uvicorn app.main:app --reload
```

Backend będzie dostępny pod adresem: **http://127.0.0.1:8000**

## Dokumentacja API

Po uruchomieniu backendu:
- **Swagger UI:** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc

## Główne endpointy

### Piosenki
```bash
GET  /                              # Status API
GET  /songs/all?limit=20            # Wszystkie utwory
GET  /songs/{tag}                   # Wyszukiwanie po tagu (np. /songs/jazz)
GET  /songs?q=query&limit=10        # Wyszukiwanie ogólne
```

### Spotify
```bash
GET  /login                         # Logowanie do Spotify
GET  /callback                      # Callback OAuth (automatyczny)
POST /create_playlist_hardcoded     # Tworzenie playlisty
```

## Rozwiązywanie problemów

### Błąd: ExecutionPolicy (Windows)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Błąd: ModuleNotFoundError
```powershell
# Upewnij się, że venv jest aktywowany
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Błąd połączenia z PostgreSQL
1. Sprawdź czy PostgreSQL jest uruchomiony
2. Zweryfikuj hasło w `.env`
3. Upewnij się, że baza `postgres` istnieje
4. Sprawdź czy tabela `spotify_tracks` została utworzona

### Błąd Spotify API
1. Sprawdź Client ID i Secret w `.env`
2. Zweryfikuj Redirect URI w Spotify Dashboard
3. Uruchom `/login` w przeglądarce

## Zależności

```
fastapi>=0.104.1          # Framework webowy
uvicorn[standard]>=0.24.0 # Serwer ASGI
sqlalchemy>=2.0.23        # ORM
psycopg2-binary>=2.9.9    # PostgreSQL adapter
python-dotenv>=1.0.0      # Zmienne środowiskowe
spotipy>=2.23.0           # Spotify API
```

## Struktura projektu

```
backend/
├── app/
│   ├── __init__.py       # Inicjalizacja pakietu
│   ├── main.py           # Endpointy FastAPI
│   ├── database.py       # Konfiguracja bazy danych
│   ├── models.py         # Model Song (SQLAlchemy)
│   └── schemas.py        # Schematy Pydantic
├── venv/                 # Środowisko wirtualne
├── .env                  # Konfiguracja (NIE COMMITOWAĆ!)
├── .env.example          # Szablon konfiguracji (commitowany)
├── .gitignore            # Wykluczenia Git
├── requirements.txt      # Zależności Python
├── populate_database.py  # Skrypt dodawania danych
├── setup_backend.ps1     # Instalacja backendu (Windows)
├── run_backend.ps1       # Uruchomienie backendu (Windows)
└── README.md            # Ta dokumentacja
```

## Bezpieczeństwo

### Ochrona wrażliwych danych

**Plik `.env` jest już chroniony:**
-  Dodany do `.gitignore` - nie będzie commitowany
-  Zawiera hasła i klucze API
-  Każdy developer ma swój lokalny plik

**Używaj `.env.example` jako szablonu:**
```powershell
# 1. Skopiuj szablon
copy .env.example .env

# 2. Edytuj .env i dodaj swoje klucze
# 3. NIE COMMITUJ pliku .env do Git!
```

**Sprawdź co jest w Git:**
```powershell
# Sprawdź czy .env NIE jest śledzony
git status

# Jeśli przypadkowo dodałeś .env do Git:
git rm --cached .env
git commit -m "Remove .env from repository"
```



