import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { memo } from 'react';
import PlaylistItem from './PlaylistItem';

interface SortablePlaylistItemProps {
  id: string;
  title: string;
  artist: string;
  duration?: string;
  onDelete: (id: string) => void;
}

const SortablePlaylistItem = memo(({ id, title, artist, duration, onDelete }: SortablePlaylistItemProps) => {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id });

  const style = {
    transform: isDragging
      ? `${CSS.Transform.toString(transform)} scale(1.02)`
      : CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.6 : 1,
    cursor: isDragging ? 'grabbing' : 'grab',
    zIndex: isDragging ? 50 : 'auto',
  };

  return (
    <div
      ref={setNodeRef}
      style={style}
      {...attributes}
      {...listeners}
      className={isDragging ? 'shadow-2xl' : ''}
    >
      <PlaylistItem id={id} title={title} artist={artist} duration={duration} onDelete={onDelete} />
    </div>
  );
});

SortablePlaylistItem.displayName = 'SortablePlaylistItem';

export default SortablePlaylistItem;
