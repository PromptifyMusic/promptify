# ===============================================
# PROMPTIFY BACKEND - INSTALACJA I KONFIGURACJA
# ===============================================
# Skrypt automatycznie przygotowuje backend do pracy
# Uzycie: .\setup_backend.ps1

param(
    [switch]$SkipPostgres = $false
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "   PROMPTIFY BACKEND - INSTALACJA" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Przejdź do katalogu backend
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# ===============================================
# KROK 1: Sprawdzenie Python
# ===============================================
Write-Host "[1/6] Sprawdzanie Python..." -ForegroundColor Yellow

try {
    $pythonVersion = python --version 2>&1
    Write-Host "  [OK] $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  [BLAD] Python nie jest zainstalowany lub nie jest w PATH" -ForegroundColor Red
    Write-Host "  [INFO] Pobierz Python z: https://www.python.org/downloads/" -ForegroundColor Gray
    exit 1
}

# ===============================================
# KROK 2: Sprawdzenie PostgreSQL
# ===============================================
if (-not $SkipPostgres) {
    Write-Host "[2/6] Sprawdzanie PostgreSQL..." -ForegroundColor Yellow

    $postgresService = Get-Service -Name "postgresql-x64-18" -ErrorAction SilentlyContinue

    if ($null -eq $postgresService) {
        # Szukaj innych wersji PostgreSQL
        $allPostgres = Get-Service | Where-Object { $_.Name -like "*postgres*" }

        if ($allPostgres.Count -eq 0) {
            Write-Host "  [!] PostgreSQL nie jest zainstalowany" -ForegroundColor Red
            Write-Host "  [INFO] Pobierz PostgreSQL z: https://www.postgresql.org/download/windows/" -ForegroundColor Gray
            $continue = Read-Host "  [?] Kontynuowac bez PostgreSQL? (t/n)"
            if ($continue -ne "t") { exit 1 }
        } else {
            Write-Host "  [OK] Znaleziono PostgreSQL: $($allPostgres[0].Name)" -ForegroundColor Green
            if ($allPostgres[0].Status -ne "Running") {
                Write-Host "  [!] PostgreSQL nie jest uruchomiony" -ForegroundColor Yellow
                $start = Read-Host "  [?] Uruchomic PostgreSQL? (t/n)"
                if ($start -eq "t") {
                    try {
                        Start-Service $allPostgres[0].Name
                        Write-Host "  [OK] PostgreSQL uruchomiony" -ForegroundColor Green
                    } catch {
                        Write-Host "  [BLAD] Nie udalo sie uruchomic PostgreSQL: $($_.Exception.Message)" -ForegroundColor Red
                        Write-Host "  [INFO] Sprobuj uruchomic recznie jako administrator" -ForegroundColor Gray
                    }
                }
            } else {
                Write-Host "  [OK] PostgreSQL dziala" -ForegroundColor Green
            }
        }
    } else {
        if ($postgresService.Status -eq "Running") {
            Write-Host "  [OK] PostgreSQL dziala" -ForegroundColor Green
        } else {
            Write-Host "  [!] PostgreSQL nie jest uruchomiony" -ForegroundColor Yellow
            $start = Read-Host "  [?] Uruchomic PostgreSQL? (t/n)"
            if ($start -eq "t") {
                try {
                    Start-Service postgresql-x64-18
                    Write-Host "  [OK] PostgreSQL uruchomiony" -ForegroundColor Green
                } catch {
                    Write-Host "  [BLAD] Nie udalo sie uruchomic PostgreSQL: $($_.Exception.Message)" -ForegroundColor Red
                    Write-Host "  [INFO] Sprobuj uruchomic recznie jako administrator" -ForegroundColor Gray
                }
            }
        }
    }
} else {
    Write-Host "[2/6] Pominieto sprawdzanie PostgreSQL (--SkipPostgres)" -ForegroundColor Gray
}

# ===============================================
# KROK 3: Tworzenie środowiska wirtualnego
# ===============================================
Write-Host "[3/6] Tworzenie srodowiska wirtualnego..." -ForegroundColor Yellow

if (Test-Path "venv") {
    Write-Host "  [!] Srodowisko wirtualne juz istnieje" -ForegroundColor Yellow
    $recreate = Read-Host "  [?] Odtworzyc od nowa? (t/n)"
    if ($recreate -eq "t") {
        try {
            Write-Host "  [*] Usuwanie starego venv..." -ForegroundColor Gray
            Remove-Item -Recurse -Force venv
            Write-Host "  [*] Tworzenie nowego venv..." -ForegroundColor Gray
            python -m venv venv
            Write-Host "  [OK] Srodowisko wirtualne utworzone" -ForegroundColor Green
        } catch {
            Write-Host "  [BLAD] Nie udalo sie odtworzyc venv: $($_.Exception.Message)" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "  [OK] Uzywanie istniejacego venv" -ForegroundColor Green
    }
} else {
    try {
        Write-Host "  [*] Tworzenie venv..." -ForegroundColor Gray
        python -m venv venv
        Write-Host "  [OK] Srodowisko wirtualne utworzone" -ForegroundColor Green
    } catch {
        Write-Host "  [BLAD] Nie udalo sie utworzyc venv: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "  [INFO] Sprawdz czy Python jest poprawnie zainstalowany" -ForegroundColor Gray
        exit 1
    }
}

# ===============================================
# KROK 4: Instalacja zależności
# ===============================================
Write-Host "[4/6] Instalacja zaleznosci Python..." -ForegroundColor Yellow

try {
    Write-Host "  [*] Aktywacja venv..." -ForegroundColor Gray
    & ".\venv\Scripts\Activate.ps1"

    Write-Host "  [*] Aktualizacja pip..." -ForegroundColor Gray
    python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [!] Ostrzezenie: Aktualizacja pip zakonczyla sie z bledami" -ForegroundColor Yellow
    } else {
        Write-Host "  [OK] pip zaktualizowany" -ForegroundColor Green
    }

    Write-Host ""
    Write-Host "  [*] Instalacja pakietow z requirements.txt..." -ForegroundColor Gray
    Write-Host "  (moze to potrwac kilka minut...)" -ForegroundColor Gray
    Write-Host ""

    pip install -r requirements.txt

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "  [BLAD] Instalacja pakietow nie powiodla sie" -ForegroundColor Red
        Write-Host "  [INFO] Sprawdz komunikaty powyzej aby poznac przyczyne" -ForegroundColor Gray
        exit 1
    }

    Write-Host ""
    Write-Host "  [OK] Wszystkie zaleznosci zainstalowane" -ForegroundColor Green
} catch {
    Write-Host "  [BLAD] Nie udalo sie zainstalowac zaleznosci" -ForegroundColor Red
    Write-Host "  [INFO] Sprawdz plik requirements.txt i polaczenie internetowe" -ForegroundColor Gray
    Write-Host "  [INFO] Szczegoly bledu: $($_.Exception.Message)" -ForegroundColor Gray
    exit 1
}

# ===============================================
# KROK 5: Konfiguracja .env
# ===============================================
Write-Host "[5/6] Konfiguracja pliku .env..." -ForegroundColor Yellow

if (Test-Path ".env") {
    Write-Host "  [OK] Plik .env juz istnieje" -ForegroundColor Green
} else {
    try {
        if (Test-Path ".env.example") {
            Write-Host "  [*] Kopiowanie .env.example do .env..." -ForegroundColor Gray
            Copy-Item ".env.example" ".env"
            Write-Host "  [OK] Plik .env utworzony" -ForegroundColor Green
            Write-Host ""
            Write-Host "  ================================================" -ForegroundColor Yellow
            Write-Host "  [!] UWAGA: Musisz edytowac plik .env!" -ForegroundColor Yellow
            Write-Host "  ================================================" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "  Uzupelnij nastepujace dane w pliku .env:" -ForegroundColor White
            Write-Host "  1. DATABASE_URL - haslo do PostgreSQL" -ForegroundColor White
            Write-Host "  2. SPOTIPY_CLIENT_ID - z Spotify Dashboard" -ForegroundColor White
            Write-Host "  3. SPOTIPY_CLIENT_SECRET - z Spotify Dashboard" -ForegroundColor White
            Write-Host ""
            Write-Host "  Link: https://developer.spotify.com/dashboard" -ForegroundColor Cyan
            Write-Host ""

            $edit = Read-Host "  [?] Otworzyc plik .env w notatniku? (t/n)"
            if ($edit -eq "t") {
                notepad .env
                Read-Host "  [?] Nacisnij Enter po zakonczonej edycji"
            }
        } else {
            Write-Host "  [!] Brak pliku .env.example - tworzenie z domyslnymi wartosciami" -ForegroundColor Yellow
            $defaultEnv = @"
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/postgres

SPOTIPY_CLIENT_ID=your_spotify_client_id_here
SPOTIPY_CLIENT_SECRET=your_spotify_client_secret_here
SPOTIPY_REDIRECT_URI=http://127.0.0.1:8000/callback
"@
            # Zapisz z UTF-8 bez BOM
            [System.IO.File]::WriteAllText("$PWD\.env", $defaultEnv, [System.Text.UTF8Encoding]::new($false))
            Write-Host "  [OK] Plik .env utworzony - UZUPELNIJ DANE!" -ForegroundColor Yellow
            notepad .env
        }
    } catch {
        Write-Host "  [BLAD] Nie udalo sie utworzyc pliku .env: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "  [INFO] Mozesz utworzyc plik .env recznie" -ForegroundColor Gray
        exit 1
    }
}

# ===============================================
# KROK 6: Sprawdzenie tabeli w bazie danych
# ===============================================
Write-Host "[6/6] Sprawdzanie bazy danych..." -ForegroundColor Yellow

$checkDb = Read-Host "  [?] Sprawdzic polaczenie z baza danych? (t/n)"
if ($checkDb -eq "t") {
    try {
        Write-Host "  [*] Testowanie polaczenia..." -ForegroundColor Gray

        # Utwórz tymczasowy plik Python
        $testScript = @"
import os, sys, traceback
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Zaladuj .env z biezacego katalogu
load_dotenv('.env')

def main():
    try:
        db_url = os.getenv('DATABASE_URL')
        if not db_url:
            print('[BLAD] DATABASE_URL nie jest ustawiony w .env', file=sys.stderr)
            sys.exit(1)
        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute(text('SELECT version();'))
            print('[OK] Polaczenie z baza danych dziala')
            try:
                result = conn.execute(text('SELECT COUNT(*) FROM spotify_tracks;'))
                count = result.fetchone()[0]
                print(f'[OK] Tabela spotify_tracks zawiera {count} utworow')
            except Exception:
                print('[INFO] Tabela spotify_tracks nie istnieje - trzeba ja utworzyc')
    except Exception:
        # Wypisz pelny traceback dla latwiejszej diagnostyki i zwroc kod bledu
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
"@

        # Zapisz do tymczasowego pliku i wykonaj (przechwyc output i kod wyjscia)
        $testScript | Out-File -FilePath "temp_test_db.py" -Encoding UTF8
        $output = python temp_test_db.py 2>&1
        $pythonExit = $LASTEXITCODE
        Write-Host $output
        Remove-Item "temp_test_db.py" -ErrorAction SilentlyContinue

        if ($pythonExit -ne 0) {
            Write-Host ""
            Write-Host "  [BLAD] Polaczenie z baza danych nie powiodlo sie" -ForegroundColor Red
            Write-Host "  [INFO] Sprawdz DATABASE_URL w pliku .env oraz szczegoly bledu powyzej" -ForegroundColor Gray
        } else {
            Write-Host ""
            $populate = Read-Host "  [?] Wstawic przykladowe dane do bazy? (t/n)"
            if ($populate -eq "t") {
                if (Test-Path "populate_database.py") {
                    Write-Host "  [*] Wstawianie danych..." -ForegroundColor Gray
                    python populate_database.py
                } else {
                    Write-Host "  [!] Brak pliku populate_database.py" -ForegroundColor Red
                }
            }
        }
    } catch {
        Write-Host "  [BLAD] Nie udalo sie przetestowac polaczenia: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "  [INFO] Sprawdz DATABASE_URL w pliku .env" -ForegroundColor Gray
    }
} else {
    Write-Host "  [*] Pominieto sprawdzanie bazy danych" -ForegroundColor Gray
}

# ===============================================
# PODSUMOWANIE
# ===============================================
Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host "   INSTALACJA ZAKONCZONA POMYSLNIE!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Nastepne kroki:" -ForegroundColor White
Write-Host "1. Sprawdz plik .env i uzupelnij dane" -ForegroundColor White
Write-Host "2. Uruchom backend: .\run_backend.ps1" -ForegroundColor White
Write-Host "3. Otworz dokumentacje: http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "Aby uruchomic backend:" -ForegroundColor Cyan
Write-Host "  .\run_backend.ps1" -ForegroundColor Yellow
Write-Host ""
