import { useState, useCallback } from 'react';
import { PlaylistItem } from '../components/playlist/PlaylistSection';
import { getRandomTrack, formatDuration } from '../services/api';

/**
 * Custom hook do zarządzania operacjami na playliście
 * Zgodny z najlepszymi praktykami React - separation of concerns
 */
export const usePlaylistOperations = () => {
    const [regeneratingItems, setRegeneratingItems] = useState<Set<string>>(new Set());
    const [isAddingItem, setIsAddingItem] = useState(false);

    /**
     * Regeneruje pojedynczy utwór w playliście
     */
    const regenerateItem = useCallback(async (
        id: string,
        onSuccess: (updatedItem: Omit<PlaylistItem, 'id'>) => void,
        onError?: (error: Error) => void
    ) => {
        setRegeneratingItems((prev) => new Set(prev).add(id));

        try {
            const track = await getRandomTrack();

            const updatedItem = {
                trackId: track.track_id,
                title: track.name,
                artist: track.artist,
                duration: formatDuration(track.duration_ms),
                image: track.image,
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
     * Dodaje nowy losowy utwór do playlisty
     */
    const addItem = useCallback(async (
        onSuccess: (newItem: Omit<PlaylistItem, 'id'>) => void,
        onError?: (error: Error) => void
    ) => {
        setIsAddingItem(true);

        try {
            const track = await getRandomTrack();

            const newItem = {
                trackId: track.track_id,
                title: track.name,
                artist: track.artist,
                duration: formatDuration(track.duration_ms),
                image: track.image,
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

