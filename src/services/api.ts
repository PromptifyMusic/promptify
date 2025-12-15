// API service for backend communication

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export interface TempPlaylistTrack {
    track_id: string;      // ID z bazy danych (Million Song Dataset)
    spotify_id: string;    // ID Spotify (używane do API)
    name: string;
    artist: string;
    duration_ms: number;
    image: string | null;
}

/**
 * Generuje tymczasową playlistę poprzez losowanie utworów z bazy danych
 */
export const generateTempPlaylist = async (
    quantity: number
): Promise<TempPlaylistTrack[]> => {
    try {
        const response = await fetch(`${API_BASE_URL}/temp_playlist_generator`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ quantity }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => null);
            throw new Error(
                errorData?.detail || `HTTP error! status: ${response.status}`
            );
        }

        return await response.json();
    } catch (error) {
        console.error('Error generating temp playlist:', error);
        throw error;
    }
};

/**
 * Pobiera jeden losowy utwór z bazy danych
 * Używane do regeneracji lub dodawania nowych utworów
 */
export const getRandomTrack = async (): Promise<TempPlaylistTrack> => {
    try {
        const response = await fetch(`${API_BASE_URL}/random_track`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => null);
            throw new Error(
                errorData?.detail || `HTTP error! status: ${response.status}`
            );
        }

        return await response.json();
    } catch (error) {
        console.error('Error getting random track:', error);
        throw error;
    }
};

/**
 * Konwertuje milisekundy na format MM:SS
 */
export const formatDuration = (durationMs: number): string => {
    const totalSeconds = Math.floor(durationMs / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
};

/**
 * Eksportuje playlistę do Spotify
 */
export interface ExportPlaylistRequest {
    name: string;
    description?: string;
    track_ids: string[];
    public?: boolean;
}

export interface ExportPlaylistResponse {
    status: string;
    message: string;
    playlist_id: string;
    playlist_url: string;
    playlist_name: string;
    tracks_count: number;
    public: boolean;
}

export const exportPlaylistToSpotify = async (
    request: ExportPlaylistRequest
): Promise<ExportPlaylistResponse> => {
    try {
        const response = await fetch(`${API_BASE_URL}/export_playlist`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(request),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => null);
            throw new Error(
                errorData?.detail || `HTTP error! status: ${response.status}`
            );
        }

        return await response.json();
    } catch (error) {
        console.error('Error exporting playlist to Spotify:', error);
        throw error;
    }
};

