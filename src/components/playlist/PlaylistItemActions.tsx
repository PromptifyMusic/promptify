import { X, RefreshCw } from 'lucide-react';
import { memo } from 'react';

interface PlaylistItemActionsProps {
  onDelete: () => void;
  onRegenerate: () => void;
  isRegenerating?: boolean;
}

const PlaylistItemActions = memo(({ onDelete, onRegenerate, isRegenerating = false }: PlaylistItemActionsProps) => {
  return (
    <div className="flex items-center gap-2 overflow-hidden transition-all duration-200 w-0 group-hover:w-[88px]">
      <button
        onClick={onRegenerate}
        disabled={isRegenerating}
        className="p-2 rounded-md bg-white/5 hover:bg-blue-500/20 text-white/60 hover:text-blue-400 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500/50 disabled:opacity-50 disabled:cursor-not-allowed opacity-0 group-hover:opacity-100"
        aria-label="Regeneruj utwór"
        type="button"
      >
        <RefreshCw className={`w-4 h-4 ${isRegenerating ? 'animate-spin' : ''}`} />
      </button>
      <button
        onClick={onDelete}
        disabled={isRegenerating}
        className="p-2 rounded-md bg-white/5 hover:bg-red-500/20 text-white/60 hover:text-red-400 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-red-500/50 disabled:opacity-50 disabled:cursor-not-allowed opacity-0 group-hover:opacity-100"
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
