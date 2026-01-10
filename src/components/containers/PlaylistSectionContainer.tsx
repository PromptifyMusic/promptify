import PlaylistSection from '../playlist/PlaylistSection';
import { usePlaylistActions } from '../../hooks/usePlaylistActions';
import { usePlaylistContext } from '../../context/PlaylistContext';

export function PlaylistSectionContainer() {
  const {
    items,
    isExpanded,
    name,
    initialQuantity,
    regeneratingItems,
    isAddingItem,
    handleReorderItems,
    handleDeleteItem,
    handleRegenerateItem,
    handleAddItem,
    handlePlaylistNameChange,
    handleCollapse,
  } = usePlaylistActions();
  const { deletingItems } = usePlaylistContext();

  return (
    <PlaylistSection
      isExpanded={isExpanded}
      playlistItems={items}
      regeneratingItems={regeneratingItems}
      deletingItems={deletingItems}
      initialQuantity={initialQuantity}
      isAddingItem={isAddingItem}
      playlistName={name}
      onCollapse={handleCollapse}
      onReorderItems={handleReorderItems}
      onDeleteItem={handleDeleteItem}
      onRegenerateItem={handleRegenerateItem}
      onAddItem={handleAddItem}
      onPlaylistNameChange={handlePlaylistNameChange}
    />
  );
}