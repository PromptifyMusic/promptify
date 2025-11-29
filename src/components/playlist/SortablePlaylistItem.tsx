import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import PlaylistItem from './PlaylistItem';

interface SortablePlaylistItemProps {
  id: string;
  title: string;
  artist: string;
  duration?: string;
}

const SortablePlaylistItem = ({ id, title, artist, duration }: SortablePlaylistItemProps) => {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.6 : 1,
    cursor: isDragging ? 'grabbing' : 'grab',
    zIndex: isDragging ? 50 : 'auto',
    scale: isDragging ? '1.02' : '1',
  };

  return (
    <div
      ref={setNodeRef}
      style={style}
      {...attributes}
      {...listeners}
      className={isDragging ? 'shadow-2xl' : ''}
    >
      <PlaylistItem title={title} artist={artist} duration={duration} />
    </div>
  );
};

export default SortablePlaylistItem;

