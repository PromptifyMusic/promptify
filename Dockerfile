

FROM python:3.10-slim

# Instalujemy narzędzia systemowe wymagane do kompilacji niektórych bibliotek (np. numpy, psycopg2)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Ustawiamy katalog roboczy kontenera
WORKDIR /app

# Kopiujemy requirements z folderu backend
COPY backend/requirements.txt .

# Instalujemy zależności (bez cache, żeby odchudzić obraz)
RUN pip install --no-cache-dir -r requirements.txt

# --- KLUCZOWE ---

RUN python -m spacy download pl_core_news_lg
RUN python -m spacy download en_core_web_md

# Kopiujemy cały kod z folderu backend do kontenera
COPY backend/ .

# Uruchamiamy aplikację
# Zwróć uwagę na ścieżkę: app.main:app (bo skopiowaliśmy zawartość backend/ bezpośrednio do /app)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]