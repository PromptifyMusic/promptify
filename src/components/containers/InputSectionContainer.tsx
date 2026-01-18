import InputSection from '../layout/InputSection';
import { usePlaylistActions } from '../../hooks/usePlaylistActions';
import { usePlaylistContext } from '../../context/PlaylistContext';

export function InputSectionContainer() {
  const { isExpanded, isLoading, handleCreatePlaylist } = usePlaylistActions();
  const { shouldClearPrompt, setShouldClearPrompt } = usePlaylistContext();

  const handlePromptCleared = () => {
    setShouldClearPrompt(false);
  };

  return (
    <InputSection
      isPlaylistExpanded={isExpanded}
      onCreatePlaylist={handleCreatePlaylist}
      isLoading={isLoading}
      shouldClearPrompt={shouldClearPrompt}
      onPromptCleared={handlePromptCleared}
    />
  );
}
