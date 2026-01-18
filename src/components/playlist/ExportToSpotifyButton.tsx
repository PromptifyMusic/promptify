import { memo, useState } from 'react';
import { Music2, ExternalLink, Info } from 'lucide-react';
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
    const [exportSuccess, setExportSuccess] = useState(false);
    const [playlistUrl, setPlaylistUrl] = useState<string | null>(null);
    const { isAuthenticated, isLoading } = useSpotifyAuth();

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

            setPlaylistUrl(response.playlist_url);
            setExportSuccess(true);
            showToast.success(`Playlista "${response.playlist_name}" została utworzona w Spotify!`);
        } catch (error) {
            console.error('Błąd podczas eksportowania do Spotify:', error);
        } finally {
            setExportingToSpotify(false);
        }
    };

    const handleOpenSpotify = () => {
        if (playlistUrl) {
            window.open(playlistUrl, '_blank', 'noopener,noreferrer');
        }
    };

    return (
        <div className="mt-6 flex flex-col items-center gap-3">
            <ActionButton
                onClick={exportSuccess ? handleOpenSpotify : handleExportToSpotify}
                loading={exportingToSpotify}
                disabled={!isAuthenticated || exportingToSpotify}
                className="action-button--spotify"
            >
                {exportSuccess ? (
                    <>
                        <ExternalLink size={20} className="inline-block align-middle mr-2" />
                        Otwórz w Spotify
                    </>
                ) : (
                    <>
                        <Music2 size={20} className="inline-block align-middle mr-2" />
                        {exportingToSpotify ? 'Eksportowanie...' : 'Eksportuj do Spotify'}
                    </>
                )}
            </ActionButton>
            {!isLoading && !isAuthenticated && (
                <div className="flex items-center gap-2 text-white/60 text-sm">
                    <Info size={16} className="text-white/50" />
                    <span>Zaloguj się do Spotify, aby wyeksportować playlistę</span>
                </div>
            )}
        </div>
    );
});

ExportToSpotifyButton.displayName = 'ExportToSpotifyButton';

export default ExportToSpotifyButton;
