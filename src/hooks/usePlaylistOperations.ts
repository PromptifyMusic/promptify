import { useState, useCallback } from 'react';
import type { PlaylistItem } from '../types';
import { replaceSong, formatDuration, extractImageUrl } from '../services/api';

/**
 * Custom hook do zarządzania operacjami na playliście
 * Zgodny z najlepszymi praktykami React - separation of concerns
 */
export const usePlaylistOperations = () => {
    const [regeneratingItems, setRegeneratingItems] = useState<Set<string>>(new Set());
    const [isAddingItem, setIsAddingItem] = useState(false);

    /**
     * Regeneruje pojedynczy utwór w playliście używając inteligentnej wymiany
     */
    const regenerateItem = useCallback(async (
        id: string,
        prompt: string,
        currentPlaylistIds: string[],
        rejectedSongId: string,
        onSuccess: (updatedItem: Omit<PlaylistItem, 'id'>) => void,
        onError?: (error: Error) => void
    ) => {
        setRegeneratingItems((prev) => new Set(prev).add(id));

        try {
            const track = await replaceSong(prompt, rejectedSongId, currentPlaylistIds);

            const updatedItem = {
                trackId: track.spotify_id,
                spotifyId: track.spotify_id,
                title: track.name,
                artist: track.artist || 'Unknown Artist',
                duration: formatDuration(track.duration_ms),
                image: extractImageUrl(track.album_images),
            };

            onSuccess(updatedItem);
        } catch (error) {
            console.error('Error during regeneration:', error);
            if (onError) {
                onError(error as Error);
            } else {
                alert('Błąd podczas regeneracji utworu. Sprawdź konsolę lub połączenie z backendem.');
            }
        } finally {
            setRegeneratingItems((prev) => {
                const newSet = new Set(prev);
                newSet.delete(id);
                return newSet;
            });
        }
    }, []);

    /**
     * Dodaje nowy utwór do playlisty (także używa inteligentnej wymiany, ale bez rejected_song_id)
     */
    const addItem = useCallback(async (
        prompt: string,
        currentPlaylistIds: string[],
        onSuccess: (newItem: Omit<PlaylistItem, 'id'>) => void,
        onError?: (error: Error) => void
    ) => {
        setIsAddingItem(true);

        try {
            // Używamy pustego rejected_song_id dla dodawania nowego utworu
            const track = await replaceSong(prompt, '', currentPlaylistIds);

            const newItem = {
                trackId: track.spotify_id,
                spotifyId: track.spotify_id,
                title: track.name,
                artist: track.artist || 'Unknown Artist',
                duration: formatDuration(track.duration_ms),
                image: extractImageUrl(track.album_images),
            };

            onSuccess(newItem);
        } catch (error) {
            console.error('Error during adding item:', error);
            if (onError) {
                onError(error as Error);
            } else {
                alert('Błąd podczas dodawania utworu. Sprawdź konsolę lub połączenie z backendem.');
            }
        } finally {
            setIsAddingItem(false);
        }
    }, []);

    return {
        regeneratingItems,
        isAddingItem,
        regenerateItem,
        addItem,
    };
};

