import { useState, useEffect, useCallback } from 'react';
import type { SpotifyAuthStatus, UseSpotifyAuthReturn } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

export const useSpotifyAuth = (): UseSpotifyAuthReturn => {
    const [authStatus, setAuthStatus] = useState<SpotifyAuthStatus>({ authenticated: false });
    const [isLoading, setIsLoading] = useState(true);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);

    const checkAuthStatus = useCallback(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/auth/status`);
            const data: SpotifyAuthStatus = await response.json();
            setAuthStatus(data);

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

    useEffect(() => {
        const params = new URLSearchParams(window.location.search);
        const authStatus = params.get('spotify_auth');

        if (authStatus === 'success') {
            window.history.replaceState({}, '', window.location.pathname);
            checkAuthStatus();
        } else if (authStatus === 'cancelled') {
            console.log('[useSpotifyAuth] Autoryzacja została anulowana przez użytkownika');
            setErrorMessage(null); // Nie pokazuj błędu, to była świadoma decyzja użytkownika
            window.history.replaceState({}, '', window.location.pathname);
        } else if (authStatus === 'error') {
            const reason = params.get('reason');
            let message: string;

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
                    message = 'Wystąpił błąd podczas autoryzacji Spotify.';
                    break;
            }

            setErrorMessage(message);
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

