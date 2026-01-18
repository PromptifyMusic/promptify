import { memo, useState } from 'react';
import { Music2 } from 'lucide-react';
import ActionButton from '../shared/ActionButton.tsx';
import type { PlaylistItem } from '../../types';
import { useSpotifyAuth } from '../../hooks/useSpotifyAuth.ts';
import { exportPlaylistToSpotify } from '../../services/api.ts';
import { showToast } from '../../utils/toast';

interface ExportToSpotifyButtonProps {
    playlistName?: string;
    playlistItems: PlaylistItem[];
}

const ExportToSpotifyButton = memo(({
    playlistName,
    playlistItems,
}: ExportToSpotifyButtonProps) => {
    const [exportingToSpotify, setExportingToSpotify] = useState(false);
    const { isAuthenticated } = useSpotifyAuth();

    const handleExportToSpotify = async () => {
        if (!isAuthenticated) {
            showToast.warning('Musisz być zalogowany do Spotify, aby wyeksportować playlistę');
            return;
        }

        if (playlistItems.length === 0) {
            showToast.warning('Playlista jest pusta');
            return;
        }

        setExportingToSpotify(true);

        try {
            const response = await exportPlaylistToSpotify({
                name: playlistName || 'Moja Playlista',
                description: 'Playlista wygenerowana przez Promptify',
                song_ids: playlistItems.map(item => item.spotifyId),  // Używamy spotifyId!
                public: false,
            });

            showToast.success(`Playlista "${response.playlist_name}" została utworzona w Spotify!`);
        } catch (error) {
            console.error('Błąd podczas eksportowania do Spotify:', error);
        } finally {
            setExportingToSpotify(false);
        }
    };


    return (
        <div className="mt-6 flex justify-center">
            <ActionButton
                onClick={handleExportToSpotify}
                loading={exportingToSpotify}
                disabled={!isAuthenticated || exportingToSpotify}
                className="action-button--spotify"
            >
                <Music2 size={20} className="inline-block align-middle mr-2" />
                {exportingToSpotify ? 'Eksportowanie...' : 'Eksportuj do Spotify'}
            </ActionButton>
        </div>
    );
});

ExportToSpotifyButton.displayName = 'ExportToSpotifyButton';

export default ExportToSpotifyButton;
