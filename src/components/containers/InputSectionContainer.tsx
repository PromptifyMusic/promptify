import InputSection from '../layout/InputSection';
import { usePlaylistActions } from '../../hooks/usePlaylistActions';
export function InputSectionContainer() {
  const { isExpanded, isLoading, handleCreatePlaylist } = usePlaylistActions();
  return (
    <InputSection
      isPlaylistExpanded={isExpanded}
      onCreatePlaylist={handleCreatePlaylist}
      isLoading={isLoading}
    />
  );
}