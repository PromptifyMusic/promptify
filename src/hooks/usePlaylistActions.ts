import { useCallback } from 'react';
import type { PlaylistItem } from '../types';
import { usePlaylistContext } from '../context/PlaylistContext';
import { usePlaylistOperations } from './usePlaylistOperations';
import { generatePlaylist, formatDuration, extractImageUrl } from '../services/api';
import { generatePlaylistItemId } from '../utils/generateId';

export function usePlaylistActions() {
  const {
    items,
    isExpanded,
    isLoading,
    name,
    originalPrompt,
    initialQuantity,
    setItems,
    addItem,
    updateItem,
    deleteItem,
    reorderItems,
    setIsExpanded,
    setIsLoading,
    setName,
    setOriginalPrompt,
    setInitialQuantity,
  } = usePlaylistContext();

  const { regeneratingItems, isAddingItem, regenerateItem, addItem: addItemOperation } = usePlaylistOperations();

  const handleCreatePlaylist = useCallback(
    async (prompt: string, quantity: number) => {
      setIsLoading(true);
      setInitialQuantity(quantity);
      setOriginalPrompt(prompt);

      try {
        const tracks = await generatePlaylist(prompt, quantity);

        const playlistItems: PlaylistItem[] = tracks.map((track) => ({
          id: generatePlaylistItemId(track.spotify_id),
          trackId: track.spotify_id,
          spotifyId: track.spotify_id,
          title: track.name,
          artist: track.artist || 'Unknown Artist',
          duration: formatDuration(track.duration_ms),
          image: extractImageUrl(track.album_images),
        }));

        setItems(playlistItems);
        setIsExpanded(true);
      } catch (error) {
        console.error('Error during playlist creation:', error);
        alert('Błąd podczas tworzenia playlisty. Sprawdź konsolę lub połączenie z backendem.');
      } finally {
        setIsLoading(false);
      }
    },
    [setIsLoading, setInitialQuantity, setOriginalPrompt, setItems, setIsExpanded]
  );

  const handleReorderItems = useCallback(
    (reorderedItems: PlaylistItem[]) => {
      reorderItems(reorderedItems);
    },
    [reorderItems]
  );

  const handleDeleteItem = useCallback(
    (id: string) => {
      deleteItem(id);
    },
    [deleteItem]
  );

  const handleRegenerateItem = useCallback(
    async (id: string) => {
      const itemToReplace = items.find((item) => item.id === id);
      if (!itemToReplace) return;

      const currentSpotifyIds = items.map((item) => item.spotifyId);

      await regenerateItem(
        id,
        originalPrompt,
        currentSpotifyIds,
        itemToReplace.spotifyId,
        (updatedTrack) => {
          updateItem(id, {
            trackId: updatedTrack.trackId,
            spotifyId: updatedTrack.spotifyId,
            title: updatedTrack.title,
            artist: updatedTrack.artist,
            duration: updatedTrack.duration,
            image: updatedTrack.image,
          });
        }
      );
    },
    [items, originalPrompt, regenerateItem, updateItem]
  );

  const handleAddItem = useCallback(async () => {
    const currentSpotifyIds = items.map((item) => item.spotifyId);

    await addItemOperation(
      originalPrompt,
      currentSpotifyIds,
      (newTrack) => {
        const newItem: PlaylistItem = {
          ...newTrack,
          id: generatePlaylistItemId(newTrack.trackId),
        };
        addItem(newItem);
      }
    );
  }, [items, originalPrompt, addItemOperation, addItem]);

  const handlePlaylistNameChange = useCallback(
    (newName: string) => {
      setName(newName);
    },
    [setName]
  );

  const handleCollapse = useCallback(() => {
    setIsExpanded(false);
  }, [setIsExpanded]);

  return {
    items,
    isExpanded,
    isLoading,
    name,
    originalPrompt,
    initialQuantity,
    regeneratingItems,
    isAddingItem,

    handleCreatePlaylist,
    handleReorderItems,
    handleDeleteItem,
    handleRegenerateItem,
    handleAddItem,
    handlePlaylistNameChange,
    handleCollapse,
  };
}

