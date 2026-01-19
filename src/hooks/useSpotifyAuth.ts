import { useState, useEffect, useCallback } from 'react';
import type { SpotifyAuthStatus, UseSpotifyAuthReturn } from '../types';
import { showToast } from '../utils/toast';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

export const useSpotifyAuth = (): UseSpotifyAuthReturn => {
    const [authStatus, setAuthStatus] = useState<SpotifyAuthStatus>({ authenticated: false });
    const [isLoading, setIsLoading] = useState(true);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);

    const checkAuthStatus = useCallback(async (showErrorToast: boolean = false) => {
        try {
            const response = await fetch(`${API_BASE_URL}/auth/status`);

            if (!response.ok) {
                setAuthStatus({ authenticated: false });
                setErrorMessage(null);
                setIsLoading(false);

                return;
            }

            const data: SpotifyAuthStatus = await response.json();
            setAuthStatus(data);

            if (data.error && data.message) {
                setErrorMessage(data.message);

                if (showErrorToast) {
                    showToast.error(data.message);
                }
            } else {
                setErrorMessage(null);
            }
        } catch (error) {
            console.warn('Cannot connect to backend:', error);
            setAuthStatus({ authenticated: false });
            setErrorMessage(null);

            if (showErrorToast) {
                showToast.error('Nie można połączyć się z backendem. Sprawdź czy serwer jest uruchomiony.');
            }
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        checkAuthStatus(false);
    }, [checkAuthStatus]);

    useEffect(() => {
        const params = new URLSearchParams(window.location.search);
        const authStatus = params.get('spotify_auth');

        if (authStatus === 'success') {
            window.history.replaceState({}, '', window.location.pathname);

            checkAuthStatus(false).then(() => {
                showToast.success('Pomyślnie połączono z kontem Spotify');
            });
        } else if (authStatus === 'cancelled') {
            console.log('[useSpotifyAuth] Autoryzacja została anulowana przez użytkownika');
            setErrorMessage(null);
            setIsLoading(false);
            showToast.info('Autoryzacja Spotify została anulowana');
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
            showToast.error(message);
            setIsLoading(false);
            window.history.replaceState({}, '', window.location.pathname);
        }
    }, [checkAuthStatus]);

    const login = useCallback(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/auth/status`);

            if (!response.ok) {
                showToast.error('Nie można połączyć się serwerem');
                return;
            }

            window.location.href = `${API_BASE_URL}/login`;
        } catch (error) {
            console.warn('Cannot connect to backend:', error);
            showToast.error('Nie można połączyć się z backendem. Sprawdź czy serwer jest uruchomiony.');
        }
    }, []);

    const logout = useCallback(async () => {
        try {
            await fetch(`${API_BASE_URL}/auth/logout`, {
                method: 'POST',
            });
            setAuthStatus({ authenticated: false });
            setErrorMessage(null);
            showToast.success('Pomyślnie wylogowano z konta Spotify');
        } catch (error) {
            console.warn('Logout error (non-critical):', error);
            setAuthStatus({ authenticated: false });
            setErrorMessage(null);
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

