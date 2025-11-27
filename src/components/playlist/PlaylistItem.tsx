import { Music } from 'lucide-react';

interface PlaylistItemProps {
  title: string;
  artist: string;
  duration?: string;
}

const PlaylistItem = ({ title, artist, duration }: PlaylistItemProps) => {
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
    </div>
  );
};

export default PlaylistItem;
