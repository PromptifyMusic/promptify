import DarkVeil from "./components/layout/animatedBackground/DarkVeil.tsx";
import QuantityInput from "./components/shared/QuantityInput";
import PromptTextarea from "./components/shared/PromptTextarea.tsx";
import ActionButton from "./components/shared/ActionButton.tsx";
import ExpandablePlaylistBox from "./components/playlist/ExpandablePlaylistBox.tsx";
import SortablePlaylistItem from "./components/playlist/SortablePlaylistItem.tsx";
import Logo from "./components/shared/Logo.tsx";
import { useState } from "react";
import {
    DndContext,
    closestCenter,
    KeyboardSensor,
    PointerSensor,
    useSensor,
    useSensors,
    DragEndEvent,
} from '@dnd-kit/core';
import {
    arrayMove,
    SortableContext,
    sortableKeyboardCoordinates,
    verticalListSortingStrategy,
} from '@dnd-kit/sortable';
import {
    restrictToVerticalAxis,
    restrictToParentElement,
} from '@dnd-kit/modifiers';

function App() {
    const [isPlaylistExpanded, setIsPlaylistExpanded] = useState(false);
    const [playlistItems, setPlaylistItems] = useState<Array<{id: string, title: string, artist: string, duration: string}>>([]);
    const [regeneratingItems, setRegeneratingItems] = useState<Set<string>>(new Set());

    const sensors = useSensors(
        useSensor(PointerSensor, {
            activationConstraint: {
                distance: 8,
            },
        }),
        useSensor(KeyboardSensor, {
            coordinateGetter: sortableKeyboardCoordinates,
        })
    );

    const handleCreatePlaylist = () => {
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
    };

    const handleDragEnd = (event: DragEndEvent) => {
        const { active, over } = event;


        if (over && active.id !== over.id) {
            setPlaylistItems((items) => {
                const oldIndex = items.findIndex((item) => item.id === active.id);
                const newIndex = items.findIndex((item) => item.id === over.id);

                return arrayMove(items, oldIndex, newIndex);
            });
        }
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
                <Logo isHidden={isPlaylistExpanded} />

                <div className="w-1/3">
                    <PromptTextarea
                        maxLength={250}
                        placeholder="Wprowadź prompt do utworzenia playlisty"
                    />
                </div>
                <div className="flex flex-col items-center gap-2">
                    <QuantityInput min={1} max={10} defaultValue={1} />
                    <span className="text-white/50 text-sm">
                        Liczba utworów w playliście
                    </span>
                </div>
                <ActionButton className='bg-white rounded-md ' onClick={handleCreatePlaylist}>
                    Utwórz playlistę
                </ActionButton>

                <div className="w-full max-w-4xl">
                    <DndContext
                        sensors={sensors}
                        collisionDetection={closestCenter}
                        onDragEnd={handleDragEnd}
                        modifiers={[restrictToVerticalAxis, restrictToParentElement]}
                    >
                        <ExpandablePlaylistBox
                            maxWidth="800px"
                            minWidth="800px"
                            maxHeight="600px"
                            isExpanded={isPlaylistExpanded}
                            onCollapse={() => setIsPlaylistExpanded(false)}
                        >
                            <SortableContext
                                items={playlistItems.map(item => item.id)}
                                strategy={verticalListSortingStrategy}
                            >
                                <div className="space-y-2">
                                    {playlistItems.map((item) => (
                                        <SortablePlaylistItem
                                            key={item.id}
                                            id={item.id}
                                            title={item.title}
                                            artist={item.artist}
                                            duration={item.duration}
                                            onDelete={handleDeleteItem}
                                            onRegenerate={handleRegenerateItem}
                                            isRegenerating={regeneratingItems.has(item.id)}
                                        />
                                    ))}
                                </div>
                            </SortableContext>
                        </ExpandablePlaylistBox>
                    </DndContext>
                </div>
            </div>
        </div>
    );
}

export default App
