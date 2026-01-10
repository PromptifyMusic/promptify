/**
 * Playlist Types
 * Typy związane z reprezentacją playlisty w aplikacji
 */

export interface PlaylistItem {
  /** Unikalny ID elementu playlisty */
  id: string;
  /** ID utworu z bazy danych */
  trackId: string;
  /** ID Spotify (używane do API Spotify) */
  spotifyId: string;
  /** Tytuł utworu */
  title: string;
  /** Wykonawca */
  artist: string;
  /** Czas trwania w formacie MM:SS */
  duration: string;
  /** URL obrazka okładki albumu */
  image?: string | null;
}

