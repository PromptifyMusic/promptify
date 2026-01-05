import { memo, useState } from 'react';
import { LogIn, LogOut, User, ChevronDown, AlertCircle } from 'lucide-react';
import { useSpotifyAuth } from '../../hooks/useSpotifyAuth.ts';
import '../../styles/SpotifyAuth.css';

const SpotifyAuth = memo(() => {
    const { isAuthenticated, user, isLoading, errorMessage, login, logout } = useSpotifyAuth();
    const [isDropdownOpen, setIsDropdownOpen] = useState(false);

    if (isLoading) {
        return (
            <div className="spotify-auth-button">
                <div className="spotify-auth-button__spinner" />
            </div>
        );
    }

    // Jeśli jest błąd, pokaż komunikat
    if (errorMessage) {
        return (
            <div className="spotify-auth-button spotify-auth-button--error">
                <div className="spotify-auth-button__error-content">
                    <AlertCircle size={18} className="spotify-auth-button__error-icon" />
                    <span className="spotify-auth-button__error-text">Błąd autoryzacji</span>
                </div>
                <div className="spotify-auth-button__error-tooltip">
                    {errorMessage}
                </div>
            </div>
        );
    }

    if (isAuthenticated && user) {
        return (
            <div className="spotify-auth-button spotify-auth-button--logged-in">
                <div className="spotify-auth-button__wrapper">
                    <button
                        className="spotify-auth-button__trigger"
                        onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                        onBlur={() => setTimeout(() => setIsDropdownOpen(false), 200)}
                    >
                        <User size={18} className="spotify-auth-button__icon" />
                        <span className="spotify-auth-button__username">
                            {user.display_name || user.id}
                        </span>
                        <ChevronDown
                            size={16}
                            className={`spotify-auth-button__chevron ${isDropdownOpen ? 'spotify-auth-button__chevron--open' : ''}`}
                        />
                    </button>

                    {isDropdownOpen && (
                        <div className="spotify-auth-button__dropdown">
                            <button
                                className="spotify-auth-button__dropdown-item"
                                onClick={() => {
                                    logout();
                                    setIsDropdownOpen(false);
                                }}
                            >
                                <LogOut size={16} />
                                <span>Wyloguj</span>
                            </button>
                        </div>
                    )}
                </div>
            </div>
        );
    }

    return (
        <button
            className="spotify-auth-button spotify-auth-button--login"
            onClick={login}
        >
            <LogIn size={18} className="spotify-auth-button__icon" />
            <span className="spotify-auth-button__text">Zaloguj przez Spotify</span>
        </button>
    );
});

SpotifyAuth.displayName = 'SpotifyAuth';

export default SpotifyAuth;

