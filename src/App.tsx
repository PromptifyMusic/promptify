import DarkVeil from "./components/layout/animatedBackground/DarkVeil.tsx";
import InputSection from "./components/layout/InputSection.tsx";
import PlaylistSection, { PlaylistItem } from "./components/playlist/PlaylistSection.tsx";
import { useState } from "react";

function App() {
    const [isPlaylistExpanded, setIsPlaylistExpanded] = useState(false);
    const [playlistItems, setPlaylistItems] = useState<PlaylistItem[]>([]);
    const [regeneratingItems, setRegeneratingItems] = useState<Set<string>>(new Set());
    const [isLoading, setIsLoading] = useState(false);


    const handleCreatePlaylist = async (prompt: string, quantity: number) => {
        setIsLoading(true);

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
