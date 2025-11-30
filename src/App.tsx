import DarkVeil from "./components/layout/animatedBackground/DarkVeil.tsx";
import InputSection from "./components/layout/InputSection.tsx";
import PlaylistSection, { PlaylistItem } from "./components/playlist/PlaylistSection.tsx";
import { useState, useRef, useEffect } from "react";

function App() {
    const [isPlaylistExpanded, setIsPlaylistExpanded] = useState(false);
    const [playlistItems, setPlaylistItems] = useState<PlaylistItem[]>([]);
    const [regeneratingItems, setRegeneratingItems] = useState<Set<string>>(new Set());
    const [deletingItems, setDeletingItems] = useState<Set<string>>(new Set());
    const [isLoading, setIsLoading] = useState(false);
    const [initialQuantity, setInitialQuantity] = useState<number>(0);
    const [isAddingItem, setIsAddingItem] = useState(false);
    const deleteTimeoutsRef = useRef<Map<string, number>>(new Map());


    const handleCreatePlaylist = async (prompt: string, quantity: number) => {
        // TODO: Use prompt for actual API call
        setIsLoading(true);
        setInitialQuantity(quantity);

        try {
            // Mock API call - 3 sekundowe opóźnienie
            await new Promise((resolve) => setTimeout(resolve, 3000));

            // Generowanie mocków w zależności od quantity
            const mockArtists = ['Artist Name 1', 'Artist Name 2', 'Artist Name 3', 'Artist Name 4', 'Artist Name 5'];
            const mockTitles = ['Song Title', 'Track', 'Hit Song', 'Music Piece', 'Melody'];

            const mockPlaylist = Array.from({ length: quantity }, (_, index) => {
                const artistIndex = index % mockArtists.length;
                const titleIndex = index % mockTitles.length;
                const minutes = Math.floor(Math.random() * 3 + 2);
                const seconds = Math.floor(Math.random() * 60).toString().padStart(2, '0');

                return {
                    id: String(index + 1),
                    title: `${mockTitles[titleIndex]} ${index + 1}`,
                    artist: mockArtists[artistIndex],
                    duration: `${minutes}:${seconds}`
                };
            });

            setPlaylistItems(mockPlaylist);
            setIsPlaylistExpanded(true);
        } catch (error) {
            console.error('Error during playlist creation:', error);
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
            setRegeneratingItems((prev) => {
                const newSet = new Set(prev);
                newSet.delete(id);
                return newSet;
            });
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
        setRegeneratingItems((prev) => new Set(prev).add(id));

        try {
            // Mock API call - 3 sekundowe opóźnienie
            await new Promise((resolve) => setTimeout(resolve, 3000));

            // Mock nowych danych
            const mockArtists = ['New Artist A', 'New Artist B', 'New Artist C', 'New Artist D', 'New Artist E'];
            const mockTitles = ['Fresh Song', 'New Track', 'Another Hit', 'Different Tune', 'Random Song'];
            const randomArtist = mockArtists[Math.floor(Math.random() * mockArtists.length)];
            const randomTitle = mockTitles[Math.floor(Math.random() * mockTitles.length)];
            const randomDuration = `${Math.floor(Math.random() * 3 + 2)}:${Math.floor(Math.random() * 60).toString().padStart(2, '0')}`;

            setPlaylistItems((items) => {
                if (!items.some((item) => item.id === id)) {
                    return items;
                }
                return items.map((item) =>
                    item.id === id
                        ? { ...item, title: randomTitle, artist: randomArtist, duration: randomDuration }
                        : item
                );
            });
        } catch (error) {
            console.error('Error during regeneration:', error);
        } finally {
            setRegeneratingItems((prev) => {
                const newSet = new Set(prev);
                newSet.delete(id);
                return newSet;
            });
        }
    };

    const handleAddItem = async () => {
        setIsAddingItem(true);

        try {
            // Mock API call - 3 sekundowe opóźnienie
            await new Promise((resolve) => setTimeout(resolve, 3000));

            // Mock nowego utworu
            const mockArtists = ['Added Artist 1', 'Added Artist 2', 'Added Artist 3', 'Added Artist 4'];
            const mockTitles = ['Added Song', 'New Addition', 'Fresh Track', 'Extra Hit'];
            const randomArtist = mockArtists[Math.floor(Math.random() * mockArtists.length)];
            const randomTitle = mockTitles[Math.floor(Math.random() * mockTitles.length)];
            const randomDuration = `${Math.floor(Math.random() * 3 + 2)}:${Math.floor(Math.random() * 60).toString().padStart(2, '0')}`;

            // Znajdź najwyższe ID i dodaj 1
            const maxId = playlistItems.reduce((max, item) => {
                const itemId = parseInt(item.id);
                return itemId > max ? itemId : max;
            }, 0);

            const newItem: PlaylistItem = {
                id: String(maxId + 1),
                title: randomTitle,
                artist: randomArtist,
                duration: randomDuration
            };

            setPlaylistItems((items) => [...items, newItem]);
        } catch (error) {
            console.error('Error during adding item:', error);
        } finally {
            setIsAddingItem(false);
        }
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
                    onCollapse={() => setIsPlaylistExpanded(false)}
                    onReorderItems={handleReorderItems}
                    onDeleteItem={handleDeleteItem}
                    onRegenerateItem={handleRegenerateItem}
                    onAddItem={handleAddItem}
                />
            </div>
        </div>
    );
}

export default App
