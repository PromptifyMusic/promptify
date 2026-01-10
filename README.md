# Promptify

Generator playlist muzycznych oparty na AI, wykorzystujący Spotify API.

## Quick Setup

### Wymagania
- Docker & Docker Compose
- Node.js 18+ (dla frontendu)
- Konto Spotify Developer (opcjonalnie, dla eksportu playlist)

### Uruchomienie

1. **Konfiguracja zmiennych środowiskowych**
   
   Utwórz plik `.env` w głównym katalogu projektu z zawartością:
   ```env
   VITE_API_BASE_URL=http://127.0.0.1:8000
   DATABASE_URL=postgresql://postgres:haslo123@localhost:5432/postgres
   DB_PASSWORD=haslo123
   
   # Spotify API (opcjonalne - tylko dla eksportu playlist)
   SPOTIPY_CLIENT_ID=twój_client_id
   SPOTIPY_CLIENT_SECRET=twój_client_secret
   SPOTIPY_REDIRECT_URI=http://127.0.0.1:8000/callback
   FRONTEND_URL=http://localhost:5173
   ```

2. **Uruchom backend + bazę danych (Docker)**
   ```bash
   docker-compose up --build
   ```
   Backend: `http://localhost:8000`  
   Swagger: `http://localhost:8000/docs`

3. **Uruchom frontend (lokalnie)**
   ```bash
   npm install
   npm run dev
   ```
   Frontend: `http://localhost:5173`

### Zatrzymanie
```bash
docker-compose down
```

---

## Uwagi
- Pierwsze uruchomienie może zająć dłuższą chwilę
- Baza danych automatycznie inicjalizowana przez `init.sql`
- Po uruchamianiu backendu należy zaczekać na komunikat `INFO: Application startup complete.` widoczny w logach konsoli Docker, aby mieć pewnoiść, że został on w pełni zainicjalizowany.
- Skrypty `setup_backend.ps1` i `run_backend.ps1` są przestarzałe - używaj Docker Compose. (Pozwalają one na lokalne uruchomienie backendu wraz z przykładową bazą. Wymagają preinstalacji POSTGRESQL i Python)
