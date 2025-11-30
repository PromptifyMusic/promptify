import {memo, useState} from 'react';
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
import { Music2 } from 'lucide-react';
import ExpandablePlaylistBox from './ExpandablePlaylistBox.tsx';
import SortablePlaylistItem from './SortablePlaylistItem.tsx';
import AddPlaylistItem from './AddPlaylistItem.tsx';
import ActionButton from '../shared/ActionButton.tsx';

export interface PlaylistItem {
    id: string;
    title: string;
    artist: string;
    duration: string;
}

interface PlaylistSectionProps {
    isExpanded: boolean;
    playlistItems: PlaylistItem[];
    regeneratingItems: Set<string>;
    deletingItems: Set<string>;
    initialQuantity: number;
    isAddingItem: boolean;
    playlistName?: string;
    onCollapse: () => void;
    onReorderItems: (items: PlaylistItem[]) => void;
    onDeleteItem: (id: string) => void;
    onRegenerateItem: (id: string) => void;
    onAddItem: () => void;
    onPlaylistNameChange?: (name: string) => void;
}

const PlaylistSection = memo(({
    isExpanded,
    playlistItems,
    regeneratingItems,
    deletingItems,
    initialQuantity,
    isAddingItem,
    playlistName,
    onCollapse,
    onReorderItems,
    onDeleteItem,
    onRegenerateItem,
    onAddItem,
    onPlaylistNameChange,
}: PlaylistSectionProps) => {
    const [exportingToSpotify, setExportingToSpotify] = useState(false);

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

    const handleDragEnd = (event: DragEndEvent) => {
        const {active, over} = event;

        if (over && active.id !== over.id) {
            const oldIndex = playlistItems.findIndex((item) => item.id === active.id);
            const newIndex = playlistItems.findIndex((item) => item.id === over.id);
            const reorderedItems = arrayMove(playlistItems, oldIndex, newIndex);
            onReorderItems(reorderedItems);
        }
    };

    const handleExportToSpotify = async () => {
        setExportingToSpotify(true);
        try {
            // Mock API call - symulacja eksportu do Spotify
            await new Promise((resolve) => setTimeout(resolve, 2000));
            console.log('Eksportowanie playlisty do Spotify:', {
                name: playlistName,
                tracks: playlistItems,
            });
            // TODO: Implementacja faktycznego eksportu do Spotify
        } catch (error) {
            console.error('Błąd podczas eksportowania do Spotify:', error);
        } finally {
            setExportingToSpotify(false);
        }
    };

    return (
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
                    isExpanded={isExpanded}
                    onCollapse={onCollapse}
                    playlistName={playlistName}
                    onPlaylistNameChange={onPlaylistNameChange}
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
                                    onDelete={onDeleteItem}
                                    onRegenerate={onRegenerateItem}
                                    isRegenerating={regeneratingItems.has(item.id)}
                                    isDeleting={deletingItems.has(item.id)}
                                />
                            ))}
                            {playlistItems.length < initialQuantity && (
                                <AddPlaylistItem
                                    onAdd={onAddItem}
                                    isAdding={isAddingItem}
                                />
                            )}
                        </div>
                    </SortableContext>
                </ExpandablePlaylistBox>
            </DndContext>

            {isExpanded && playlistItems.length > 0 && (
                <div className="mt-6 flex justify-center">
                    <ActionButton
                        onClick={handleExportToSpotify}
                        loading={exportingToSpotify}
                        disabled={exportingToSpotify}
                        className="action-button--spotify"
                    >
                        <Music2 size={20} style={{ display: 'inline-block', verticalAlign: 'middle', marginRight: '8px' }} />
                        Eksportuj do Spotify
                    </ActionButton>
                </div>
            )}
        </div>
    );
});

PlaylistSection.displayName = 'PlaylistSection';

export default PlaylistSection;

