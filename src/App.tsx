import DarkVeil from "./components/layout/animatedBackground/DarkVeil.tsx";
import InputSection from "./components/layout/InputSection.tsx";
import PlaylistSection from "./components/playlist/PlaylistSection.tsx";
import type { PlaylistItem } from "./types";
import { useState, useRef, useEffect } from "react";
import SpotifyAuth from "./components/playlist/SpotifyAuth.tsx";
import { generatePlaylist, formatDuration, extractImageUrl } from "./services/api.ts";
import { usePlaylistOperations } from "./hooks/usePlaylistOperations.ts";
import { generatePlaylistItemId } from "./utils/generateId.ts";
function App() {
    const [isPlaylistExpanded, setIsPlaylistExpanded] = useState(false);
    const [playlistItems, setPlaylistItems] = useState<PlaylistItem[]>([]);
    const [deletingItems, setDeletingItems] = useState<Set<string>>(new Set());
    const [isLoading, setIsLoading] = useState(false);
    const [initialQuantity, setInitialQuantity] = useState<number>(0);
    const [playlistName, setPlaylistName] = useState<string>('Playlista');
    const [originalPrompt, setOriginalPrompt] = useState<string>('');
    const deleteTimeoutsRef = useRef<Map<string, number>>(new Map());

    // Custom hook do zarządzania operacjami na playliście
    const { regeneratingItems, isAddingItem, regenerateItem, addItem } = usePlaylistOperations();


    const handleCreatePlaylist = async (prompt: string, quantity: number) => {
        setIsLoading(true);
        setInitialQuantity(quantity);
        setOriginalPrompt(prompt); // Zapisz prompt dla przyszłych operacji

        try {
            // Wywołanie prawdziwego API backendu z promptem
            const tracks = await generatePlaylist(prompt, quantity);

            // Mapowanie danych z backendu na format PlaylistItem
            const playlistItems: PlaylistItem[] = tracks.map((track) => ({
                id: generatePlaylistItemId(track.spotify_id),  // Unikalny ID: spotifyId + timestamp
                trackId: track.spotify_id,                     // Używamy spotify_id jako trackId
                spotifyId: track.spotify_id,                   // ID Spotify
                title: track.name,
                artist: track.artist || 'Unknown Artist',
                duration: formatDuration(track.duration_ms),
                image: extractImageUrl(track.album_images),    // Parsuj JSON i wyciągnij URL
            }));

            setPlaylistItems(playlistItems);
            setIsPlaylistExpanded(true);
        } catch (error) {
            console.error('Error during playlist creation:', error);
            alert('Błąd podczas tworzenia playlisty. Sprawdź konsolę lub połączenie z backendem.');
        } finally {
            setIsLoading(false);
        }
    };

    const handleReorderItems = (reorderedItems: PlaylistItem[]) => {
        setPlaylistItems(reorderedItems);
    };

    const handleDeleteItem = (id: string) => {
        // Prevent multiple delete operations on the same item
        if (deletingItems.has(id)) {
            return;
        }

        // Oznacz element jako usuwany (rozpocznij animację)
        setDeletingItems((prev) => new Set(prev).add(id));

        // Usuń element po zakończeniu animacji (300ms)
        const timeoutId = setTimeout(() => {
            setPlaylistItems((items) => items.filter((item) => item.id !== id));
            setDeletingItems((prev) => {
                const newSet = new Set(prev);
                newSet.delete(id);
                return newSet;
            });
            deleteTimeoutsRef.current.delete(id);
        }, 300);

        deleteTimeoutsRef.current.set(id, timeoutId);
    };

    useEffect(() => {
        return () => {
            // Wyczyść wszystkie aktywne timeout'y przy unmount
            deleteTimeoutsRef.current.forEach((timeoutId) => {
                clearTimeout(timeoutId);
            });
            deleteTimeoutsRef.current.clear();
        };
    }, []);

    const handleRegenerateItem = async (id: string) => {
        // Znajdź spotify_id utworu, który ma być wymieniony
        const itemToReplace = playlistItems.find(item => item.id === id);
        if (!itemToReplace) return;

        // Pobierz listę wszystkich spotify_id z playlisty
        const currentSpotifyIds = playlistItems.map(item => item.spotifyId);

        await regenerateItem(
            id,
            originalPrompt,
            currentSpotifyIds,
            itemToReplace.spotifyId,
            (updatedTrack) => {
                setPlaylistItems((items) => {
                    if (!items.some((item) => item.id === id)) {
                        return items;
                    }
                    return items.map((item) =>
                        item.id === id
                            ? {
                                ...item,                        // Zachowaj istniejący id
                                trackId: updatedTrack.trackId,
                                spotifyId: updatedTrack.spotifyId,
                                title: updatedTrack.title,
                                artist: updatedTrack.artist,
                                duration: updatedTrack.duration,
                                image: updatedTrack.image,
                            }
                            : item
                    );
                });
            }
        );
    };

    const handleAddItem = async () => {
        // Pobierz listę wszystkich spotify_id z playlisty
        const currentSpotifyIds = playlistItems.map(item => item.spotifyId);

        await addItem(
            originalPrompt,
            currentSpotifyIds,
            (newTrack) => {
                const newItem: PlaylistItem = {
                    ...newTrack,
                    id: generatePlaylistItemId(newTrack.trackId),  // Unikalny ID
                };
                setPlaylistItems((items) => [...items, newItem]);
            }
        );
    };

    const handlePlaylistNameChange = (name: string) => {
        setPlaylistName(name);
    };

    return (
        <div className="relative w-full h-screen overflow-hidden">
            <div className="absolute inset-0 -z-10">
                <DarkVeil 
                    hueShift={180}
                    speed={0.25}
                    warpAmount={1}
                    resolutionScale={1}
                />
            </div>

            <SpotifyAuth />

            <div className="relative z-10 w-full h-full flex flex-col items-center justify-center gap-8 p-8">
                <InputSection
                    isPlaylistExpanded={isPlaylistExpanded}
                    onCreatePlaylist={handleCreatePlaylist}
                    isLoading={isLoading}
                />
                <PlaylistSection
                    isExpanded={isPlaylistExpanded}
                    playlistItems={playlistItems}
                    regeneratingItems={regeneratingItems}
                    deletingItems={deletingItems}
                    initialQuantity={initialQuantity}
                    isAddingItem={isAddingItem}
                    playlistName={playlistName}
                    onCollapse={() => setIsPlaylistExpanded(false)}
                    onReorderItems={handleReorderItems}
                    onDeleteItem={handleDeleteItem}
                    onRegenerateItem={handleRegenerateItem}
                    onAddItem={handleAddItem}
                    onPlaylistNameChange={handlePlaylistNameChange}
                />
            </div>
        </div>
    );
}

export default App
