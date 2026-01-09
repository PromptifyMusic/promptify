/**
 * Generuje unikalny identyfikator dla elementów playlisty
 * Używa trackId + timestamp dla prostoty i unikalności
 */
export const generatePlaylistItemId = (trackId: string): string => {
    return `${trackId}-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`;
};

