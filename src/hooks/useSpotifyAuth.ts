import { useState, useEffect, useCallback } from 'react';

const API_BASE_URL = 'http://127.0.0.1:8000';

interface SpotifyUser {
    id: string;
    display_name: string;
    email?: string;
}

interface AuthStatus {
    authenticated: boolean;
    user?: SpotifyUser;
    error?: string;
    message?: string;
}

export const useSpotifyAuth = () => {
    const [authStatus, setAuthStatus] = useState<AuthStatus>({ authenticated: false });
    const [isLoading, setIsLoading] = useState(true);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);

    const checkAuthStatus = useCallback(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/auth/status`);
            const data: AuthStatus = await response.json();
            setAuthStatus(data);

            // Jeśli jest komunikat błędu, pokaż go użytkownikowi
            if (data.error && data.message) {
                setErrorMessage(data.message);
            } else {
                setErrorMessage(null);
            }
        } catch (error) {
            console.error('Error checking auth status:', error);
            setAuthStatus({ authenticated: false });
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        checkAuthStatus();
    }, [checkAuthStatus]);

    // Sprawdź czy callback z Spotify był udany lub zwrócił błąd
    useEffect(() => {
        const params = new URLSearchParams(window.location.search);
        const authStatus = params.get('spotify_auth');

        if (authStatus === 'success') {
            // Usuń parametr z URL
            window.history.replaceState({}, '', window.location.pathname);
            // Odśwież status autoryzacji
            checkAuthStatus();
        } else if (authStatus === 'cancelled') {
            // Użytkownik anulował autoryzację
            console.log('[useSpotifyAuth] Autoryzacja została anulowana przez użytkownika');
            setErrorMessage(null); // Nie pokazuj błędu, to była świadoma decyzja użytkownika
            // Usuń parametry z URL
            window.history.replaceState({}, '', window.location.pathname);
        } else if (authStatus === 'error') {
            const reason = params.get('reason');
            let message = 'Wystąpił błąd podczas autoryzacji Spotify.';

            switch (reason) {
                case 'invalid_client':
                    message = 'Niepoprawne dane klienta Spotify (Client ID lub Client Secret). Sprawdź plik .env w backendzie.';
                    break;
                case 'config':
                    message = 'Błąd konfiguracji backendu. Sprawdź czy wszystkie zmienne środowiskowe są ustawione w pliku .env.';
                    break;
                case 'token_failed':
                    message = 'Nie udało się uzyskać tokena dostępu od Spotify. Spróbuj ponownie.';
                    break;
                case 'no_code':
                    message = 'Nie otrzymano kodu autoryzacyjnego od Spotify. Spróbuj ponownie.';
                    break;
                default:
                    message = 'Nieznany błąd podczas autoryzacji. Sprawdź logi backendu.';
            }

            setErrorMessage(message);
            // Usuń parametry z URL
            window.history.replaceState({}, '', window.location.pathname);
        }
    }, [checkAuthStatus]);

    const login = useCallback(() => {
        window.location.href = `${API_BASE_URL}/login`;
    }, []);

    const logout = useCallback(async () => {
        try {
            await fetch(`${API_BASE_URL}/auth/logout`, {
                method: 'POST',
            });
            setAuthStatus({ authenticated: false });
            setErrorMessage(null);
        } catch (error) {
            console.error('Error during logout:', error);
        }
    }, []);

    return {
        isAuthenticated: authStatus.authenticated,
        user: authStatus.user,
        isLoading,
        errorMessage,
        login,
        logout,
        refreshAuthStatus: checkAuthStatus,
    };
};

