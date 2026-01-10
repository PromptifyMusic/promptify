/**
 * API Types
 * Typy dla komunikacji z backendem
 */

/**
 * Reprezentacja utworu zwracana z API backendu
 */
export interface ApiPlaylistTrack {
  spotify_id: string;
  name: string;
  artist: string;
  popularity?: number;
  score?: number;
  spotify_preview_url?: string | null;
  album_images?: string | null;
  duration_ms: number;
}

/**
 * Request body dla generowania playlisty
 */
export interface GeneratePlaylistRequest {
  text: string;
  top_n: number;
}

/**
 * Request body dla zamiany utworu
 */
export interface ReplaceSongRequest {
  text: string;
  rejected_song_id: string;
  current_playlist_ids: string[];
}

/**
 * Request body dla eksportu playlisty do Spotify
 */
export interface ExportPlaylistRequest {
  name: string;
  description?: string;
  song_ids: string[];
  public?: boolean;
}

/**
 * Response z API przy eksporcie playlisty
 */
export interface ExportPlaylistResponse {
  status: string;
  message: string;
  playlist_id: string;
  playlist_url: string;
  playlist_name: string;
  tracks_count: number;
  public: boolean;
}

/**
 * Generyczny typ dla błędów API
 */
export interface ApiError {
  detail?: string;
  message?: string;
  status?: number;
}

/**
 * @deprecated - Legacy type, zostawiony dla kompatybilności wstecznej
 * Stara reprezentacja utworu z tymczasowego API
 */
export interface TempPlaylistTrack {
  track_id: string;
  spotify_id: string;
  name: string;
  artist: string;
  duration_ms: number;
  image: string | null;
}

