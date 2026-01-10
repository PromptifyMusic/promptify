/**
 * Spotify Types
 * Typy związane z integracją Spotify
 */

/**
 * Reprezentacja użytkownika Spotify
 */
export interface SpotifyUser {
  id: string;
  display_name: string;
  email?: string;
  images?: SpotifyImage[];
  country?: string;
  product?: 'free' | 'premium';
}

/**
 * Obraz Spotify (avatar, okładka albumu, itp.)
 */
export interface SpotifyImage {
  url: string;
  width: number | null;
  height: number | null;
}

/**
 * Status autoryzacji Spotify
 */
export interface SpotifyAuthStatus {
  authenticated: boolean;
  user?: SpotifyUser;
  error?: string;
  message?: string;
}

/**
 * Hook return type dla useSpotifyAuth
 */
export interface UseSpotifyAuthReturn {
  isAuthenticated: boolean;
  user?: SpotifyUser;
  isLoading: boolean;
  errorMessage: string | null;
  login: () => void;
  logout: () => Promise<void>;
  refreshAuthStatus: () => Promise<void>;
}

