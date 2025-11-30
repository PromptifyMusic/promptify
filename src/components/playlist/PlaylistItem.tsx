import { Music } from 'lucide-react';
import { memo } from 'react';
import PlaylistItemActions from './PlaylistItemActions';

interface PlaylistItemProps {
  id: string;
  title: string;
  artist: string;
  duration?: string;
  onDelete: (id: string) => void;
  onRegenerate: (id: string) => void;
  isRegenerating?: boolean;
  isDeleting?: boolean;
}

const PlaylistItem = memo(({ id, title, artist, duration, onDelete, onRegenerate, isRegenerating = false, isDeleting = false }: PlaylistItemProps) => {
  const handleDelete = () => {
    onDelete(id);
  };

  const handleRegenerate = () => {
    onRegenerate(id);
  };

  return (
    <div className={`group flex items-center gap-4 p-3 rounded-lg hover:bg-white/5 transition-all duration-200 cursor-pointer ${
      isRegenerating ? 'animate-pulse bg-white/5' : ''
    } ${isDeleting ? 'scale-95 blur-sm' : ''}`}>
      <div className={`flex-shrink-0 w-12 h-12 bg-white/10 rounded-md flex items-center justify-center transition-opacity duration-200 ${
        isRegenerating ? 'opacity-50' : ''
      }`}>
        <Music className="w-6 h-6 text-white/70" />
      </div>

      <div className={`flex-1 min-w-0 transition-opacity duration-200 ${
        isRegenerating ? 'opacity-50' : ''
      }`}>
        <h4 className="text-white font-medium truncate">{title}</h4>
        <p className="text-white/60 text-sm truncate">{artist}</p>
      </div>

      <div className="flex items-center gap-1">
        {duration && (
          <div className={`flex-shrink-0 text-white/50 text-sm transition-all duration-200 group-hover:translate-x-0 ${
            isRegenerating ? 'opacity-50' : ''
          }`}>
            {duration}
          </div>
        )}
        <div className="ml-5">
            <PlaylistItemActions
                onDelete={handleDelete}
                onRegenerate={handleRegenerate}
                isRegenerating={isRegenerating}
            />
        </div>
      </div>
    </div>
  );
});

PlaylistItem.displayName = 'PlaylistItem';

export default PlaylistItem;
