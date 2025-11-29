import { X } from 'lucide-react';
import { memo } from 'react';

interface PlaylistItemActionsProps {
  onDelete: () => void;
}

const PlaylistItemActions = memo(({ onDelete }: PlaylistItemActionsProps) => {
  return (
    <div className="flex-shrink-0 flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
      <button
        onClick={onDelete}
        className="p-2 rounded-md bg-white/5 hover:bg-red-500/20 text-white/60 hover:text-red-400 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-red-500/50"
        aria-label="Usuń utwór"
        type="button"
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  );
});

PlaylistItemActions.displayName = 'PlaylistItemActions';

export default PlaylistItemActions;

