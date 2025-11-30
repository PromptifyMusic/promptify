import DarkVeil from "./components/layout/animatedBackground/DarkVeil.tsx";
import InputSection from "./components/layout/InputSection.tsx";
import PlaylistSection, { PlaylistItem } from "./components/playlist/PlaylistSection.tsx";
import { useState } from "react";

function App() {
    const [isPlaylistExpanded, setIsPlaylistExpanded] = useState(false);
    const [playlistItems, setPlaylistItems] = useState<PlaylistItem[]>([]);
    const [regeneratingItems, setRegeneratingItems] = useState<Set<string>>(new Set());
    const [isLoading, setIsLoading] = useState(false);


    const handleCreatePlaylist = async () => {
        setIsLoading(true);

        try {
            // Mock API call - 3 sekundowe opóźnienie
            await new Promise((resolve) => setTimeout(resolve, 3000));

            const mockPlaylist = [
                { id: "1", title: "Song Title 1", artist: "Artist Name 1", duration: "3:45" },
                { id: "2", title: "Song Title 2", artist: "Artist Name 2", duration: "4:12" },
                { id: "3", title: "Song Title 3", artist: "Artist Name 3", duration: "3:28" },
                { id: "4", title: "Song Title 4", artist: "Artist Name 4", duration: "5:01" },
                { id: "5", title: "Song Title 5", artist: "Artist Name 5", duration: "3:56" },
                { id: "6", title: "Song Title 6", artist: "Artist Name 1", duration: "3:45" },
                { id: "7", title: "Song Title 7", artist: "Artist Name 2", duration: "4:12" },
                { id: "8", title: "Song Title 8", artist: "Artist Name 3", duration: "3:28" },
                { id: "9", title: "Song Title 9", artist: "Artist Name 4", duration: "5:01" },
                { id: "10", title: "Song Title 10", artist: "Artist Name 5", duration: "3:56" },
            ];
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
        setPlaylistItems((items) => items.filter((item) => item.id !== id));
        setRegeneratingItems((prev) => {
            const newSet = new Set(prev);
            newSet.delete(id);
            return newSet;
        });
    };

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
                    onCollapse={() => setIsPlaylistExpanded(false)}
                    onReorderItems={handleReorderItems}
                    onDeleteItem={handleDeleteItem}
                    onRegenerateItem={handleRegenerateItem}
                />
            </div>
        </div>
    );
}

export default App
