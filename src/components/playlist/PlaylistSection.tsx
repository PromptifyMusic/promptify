import { memo } from 'react';
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
import ExpandablePlaylistBox from './ExpandablePlaylistBox';
import SortablePlaylistItem from './SortablePlaylistItem';

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
    onCollapse: () => void;
    onReorderItems: (items: PlaylistItem[]) => void;
    onDeleteItem: (id: string) => void;
    onRegenerateItem: (id: string) => void;
}

const PlaylistSection = memo(({
    isExpanded,
    playlistItems,
    regeneratingItems,
    onCollapse,
    onReorderItems,
    onDeleteItem,
    onRegenerateItem,
}: PlaylistSectionProps) => {
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
        const { active, over } = event;

        if (over && active.id !== over.id) {
            const oldIndex = playlistItems.findIndex((item) => item.id === active.id);
            const newIndex = playlistItems.findIndex((item) => item.id === over.id);
            const reorderedItems = arrayMove(playlistItems, oldIndex, newIndex);
            onReorderItems(reorderedItems);
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
                                />
                            ))}
                        </div>
                    </SortableContext>
                </ExpandablePlaylistBox>
            </DndContext>
        </div>
    );
});

PlaylistSection.displayName = 'PlaylistSection';

export default PlaylistSection;

