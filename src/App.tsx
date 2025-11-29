import DarkVeil from "./components/layout/animatedBackground/DarkVeil.tsx";
import QuantityInput from "./components/shared/QuantityInput";
import PromptTextarea from "./components/shared/PromptTextarea.tsx";
import ActionButton from "./components/shared/ActionButton.tsx";
import ExpandablePlaylistBox from "./components/playlist/ExpandablePlaylistBox.tsx";
import PlaylistItem from "./components/playlist/PlaylistItem.tsx";
import { useState } from "react";

function App() {
    const [isPlaylistExpanded, setIsPlaylistExpanded] = useState(false);
    const [playlistItems, setPlaylistItems] = useState<Array<{title: string, artist: string, duration: string}>>([]);

    const handleCreatePlaylist = () => {
        const mockPlaylist = [
            { title: "Song Title 1", artist: "Artist Name 1", duration: "3:45" },
            { title: "Song Title 2", artist: "Artist Name 2", duration: "4:12" },
            { title: "Song Title 3", artist: "Artist Name 3", duration: "3:28" },
            { title: "Song Title 4", artist: "Artist Name 4", duration: "5:01" },
            { title: "Song Title 5", artist: "Artist Name 5", duration: "3:56" },
            { title: "Song Title 6", artist: "Artist Name 1", duration: "3:45" },
            { title: "Song Title 7", artist: "Artist Name 2", duration: "4:12" },
            { title: "Song Title 8", artist: "Artist Name 3", duration: "3:28" },
            { title: "Song Title 9", artist: "Artist Name 4", duration: "5:01" },
            { title: "Song Title 10", artist: "Artist Name 5", duration: "3:56" },
        ];
        setPlaylistItems(mockPlaylist);
        setIsPlaylistExpanded(true);
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
                <div className="w-1/3">
                    <PromptTextarea
                        maxLength={250}
                        placeholder="Wprowadź prompt do utworzenia playlisty"
                    />
                </div>
                <QuantityInput min={1} max={10} defaultValue={1} />
                <ActionButton className='bg-white rounded-md ' onClick={handleCreatePlaylist}>
                    Utwórz playlistę
                </ActionButton>

                <div className="w-full max-w-4xl">
                    <ExpandablePlaylistBox
                        maxWidth="800px"
                        maxHeight="600px"
                        isExpanded={isPlaylistExpanded}
                        onCollapse={() => setIsPlaylistExpanded(false)}
                    >
                        <div className="space-y-2">
                            {playlistItems.map((item, index) => (
                                <PlaylistItem
                                    key={index}
                                    title={item.title}
                                    artist={item.artist}
                                    duration={item.duration}
                                />
                            ))}
                        </div>
                    </ExpandablePlaylistBox>
                </div>
            </div>
        </div>
    );
}

export default App
