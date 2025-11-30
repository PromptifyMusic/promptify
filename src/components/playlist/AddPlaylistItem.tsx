import { Plus } from 'lucide-react';
import { memo } from 'react';

interface AddPlaylistItemProps {
  onAdd: () => void;
  isAdding?: boolean;
}

const AddPlaylistItem = memo(({ onAdd, isAdding = false }: AddPlaylistItemProps) => {
  return (
    <button
      onClick={onAdd}
      disabled={isAdding}
      className={`group flex items-center gap-4 p-3 rounded-lg border-2 border-dashed border-white/20 hover:border-white/40 hover:bg-white/5 transition-all duration-200 cursor-pointer w-full ${
        isAdding ? 'animate-pulse bg-white/5 opacity-50' : ''
      }`}
    >
      <div className="flex-shrink-0 w-12 h-12 bg-white/10 rounded-md flex items-center justify-center transition-colors duration-200 group-hover:bg-white/15">
        <Plus className="w-6 h-6 text-white/70" />
      </div>

      <div className="flex-1 min-w-0 text-left">
        <h4 className="text-white/70 font-medium">
          {isAdding ? 'Dodawanie utworu...' : 'Dodaj nowy utwór'}
        </h4>
        <p className="text-white/40 text-sm">
          Wygeneruj kolejny utwór dla playlisty
        </p>
      </div>
    </button>
  );
});

AddPlaylistItem.displayName = 'AddPlaylistItem';

export default AddPlaylistItem;

