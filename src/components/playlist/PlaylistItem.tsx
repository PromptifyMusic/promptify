import { Music } from 'lucide-react';
import { memo } from 'react';
import PlaylistItemActions from './PlaylistItemActions';

interface PlaylistItemProps {
  id: string;
  title: string;
  artist: string;
  duration?: string;
  onDelete: (id: string) => void;
}

const PlaylistItem = memo(({ id, title, artist, duration, onDelete }: PlaylistItemProps) => {
  const handleDelete = () => {
    onDelete(id);
  };

  return (
    <div className="group flex items-center gap-4 p-3 rounded-lg hover:bg-white/5 transition-colors duration-200 cursor-pointer">
      <div className="flex-shrink-0 w-12 h-12 bg-white/10 rounded-md flex items-center justify-center">
        <Music className="w-6 h-6 text-white/70" />
      </div>

      <div className="flex-1 min-w-0">
        <h4 className="text-white font-medium truncate">{title}</h4>
        <p className="text-white/60 text-sm truncate">{artist}</p>
      </div>

      {duration && (
        <div className="flex-shrink-0 text-white/50 text-sm">
          {duration}
        </div>
      )}

      <PlaylistItemActions onDelete={handleDelete} />
    </div>
  );
});

PlaylistItem.displayName = 'PlaylistItem';

export default PlaylistItem;
