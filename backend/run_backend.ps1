# ===============================================
# PROMPTIFY BACKEND - URUCHOMIENIE
# ===============================================
# Szybkie uruchomienie backendu FastAPI
# Uzycie: .\run_backend.ps1

param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$NoReload = $false
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "   PROMPTIFY BACKEND - URUCHAMIANIE" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Przejdź do katalogu backend
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# ===============================================
# Sprawdzenie środowiska wirtualnego
# ===============================================
Write-Host "[1/4] Sprawdzanie srodowiska wirtualnego..." -ForegroundColor Yellow

if (-not (Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Host "  [BLAD] Brak srodowiska wirtualnego!" -ForegroundColor Red
    Write-Host "  [INFO] Uruchom najpierw: .\setup_backend.ps1" -ForegroundColor Gray
    exit 1
}

Write-Host "  [*] Aktywacja venv..." -ForegroundColor Gray
& ".\venv\Scripts\Activate.ps1"
Write-Host "  [OK] Srodowisko wirtualne aktywowane" -ForegroundColor Green

# ===============================================
# Sprawdzenie pliku .env
# ===============================================
Write-Host "[2/4] Sprawdzanie konfiguracji..." -ForegroundColor Yellow

if (-not (Test-Path ".env")) {
    Write-Host "  [BLAD] Brak pliku .env!" -ForegroundColor Red
    Write-Host "  [INFO] Uruchom najpierw: .\setup_backend.ps1" -ForegroundColor Gray
    exit 1
}

Write-Host "  [OK] Plik .env istnieje" -ForegroundColor Green

# ===============================================
# Sprawdzenie PostgreSQL
# ===============================================
Write-Host "[3/4] Sprawdzanie PostgreSQL..." -ForegroundColor Yellow

$postgresRunning = $false
$postgresService = Get-Service -Name "postgresql-x64-18" -ErrorAction SilentlyContinue

if ($null -eq $postgresService) {
    # Szukaj innych wersji
    $allPostgres = Get-Service | Where-Object { $_.Name -like "*postgres*" }
    if ($allPostgres.Count -gt 0) {
        $postgresService = $allPostgres[0]
    }
}

if ($null -ne $postgresService) {
    if ($postgresService.Status -eq "Running") {
        Write-Host "  [OK] PostgreSQL dziala ($($postgresService.Name))" -ForegroundColor Green
        $postgresRunning = $true
    } else {
        Write-Host "  [!] PostgreSQL nie jest uruchomiony" -ForegroundColor Yellow
        $start = Read-Host "  [?] Uruchomic PostgreSQL? (t/n)"
        if ($start -eq "t") {
            try {
                Start-Service $postgresService.Name
                Write-Host "  [OK] PostgreSQL uruchomiony" -ForegroundColor Green
                $postgresRunning = $true
            } catch {
                Write-Host "  [BLAD] Nie udalo sie uruchomic PostgreSQL" -ForegroundColor Red
            }
        }
    }
} else {
    Write-Host "  [!] PostgreSQL nie znaleziony" -ForegroundColor Yellow
    Write-Host "  [INFO] Backend moze nie dzialac poprawnie bez bazy danych" -ForegroundColor Gray
}

if (-not $postgresRunning) {
    $continue = Read-Host "  [?] Kontynuowac mimo to? (t/n)"
    if ($continue -ne "t") {
        exit 1
    }
}

# ===============================================
# Uruchomienie serwera
# ===============================================
Write-Host "[4/4] Uruchamianie serwera FastAPI..." -ForegroundColor Yellow
Write-Host ""

# Parametry uruchomienia
$reloadParam = if ($NoReload) { "" } else { "--reload" }

Write-Host "================================================" -ForegroundColor Green
Write-Host "   BACKEND URUCHOMIONY POMYSLNIE!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Serwer:       http://${HostAddress}:${Port}" -ForegroundColor Cyan
Write-Host "  Dokumentacja: http://${HostAddress}:${Port}/docs" -ForegroundColor Cyan
Write-Host "  ReDoc:        http://${HostAddress}:${Port}/redoc" -ForegroundColor Cyan
Write-Host ""
Write-Host "================================================" -ForegroundColor White
Write-Host ""
Write-Host "Endpointy API:" -ForegroundColor White
Write-Host "  GET  /                     - Status API" -ForegroundColor Gray
Write-Host "  GET  /songs/all            - Wszystkie utwory" -ForegroundColor Gray
Write-Host "  GET  /songs/{tag}          - Wyszukiwanie po tagu" -ForegroundColor Gray
Write-Host "  GET  /songs?q=query        - Wyszukiwanie ogolne" -ForegroundColor Gray
Write-Host "  GET  /login                - Logowanie Spotify" -ForegroundColor Gray
Write-Host ""
Write-Host "================================================" -ForegroundColor White
Write-Host ""
Write-Host "Nacisnij Ctrl+C aby zatrzymac serwer" -ForegroundColor Yellow
Write-Host ""

# Uruchom serwer
try {
    if ($NoReload) {
        python -m uvicorn app.main:app --host $HostAddress --port $Port
    } else {
        python -m uvicorn app.main:app --reload --host $HostAddress --port $Port
    }
} catch {
    Write-Host ""
    Write-Host "================================================" -ForegroundColor Red
    Write-Host "   SERWER ZATRZYMANY" -ForegroundColor Red
    Write-Host "================================================" -ForegroundColor Red
    Write-Host ""
}

