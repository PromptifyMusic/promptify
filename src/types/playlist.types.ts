/**
 * Playlist Types
 * Typy związane z reprezentacją playlisty w aplikacji
 */

/**
 * Element playlisty w UI aplikacji
 * Używany do renderowania i zarządzania playlistą
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

/**
 * Stan playlisty
 */
export interface PlaylistState {
  items: PlaylistItem[];
  isExpanded: boolean;
  isLoading: boolean;
  name: string;
  originalPrompt: string;
  initialQuantity: number;
}

/**
 * Akcje możliwe do wykonania na playliście
 */
export type PlaylistAction =
  | { type: 'SET_ITEMS'; payload: PlaylistItem[] }
  | { type: 'ADD_ITEM'; payload: PlaylistItem }
  | { type: 'UPDATE_ITEM'; payload: { id: string; item: Partial<PlaylistItem> } }
  | { type: 'DELETE_ITEM'; payload: string }
  | { type: 'REORDER_ITEMS'; payload: PlaylistItem[] }
  | { type: 'EXPAND' }
  | { type: 'COLLAPSE' }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_NAME'; payload: string }
  | { type: 'SET_PROMPT'; payload: string }
  | { type: 'SET_QUANTITY'; payload: number }
  | { type: 'RESET' };
