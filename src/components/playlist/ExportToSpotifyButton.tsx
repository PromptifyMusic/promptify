import { memo, useState } from 'react';
import { Music2, ExternalLink, CheckCircle } from 'lucide-react';
import ActionButton from '../shared/ActionButton.tsx';
import { PlaylistItem } from './PlaylistSection.tsx';
import { useSpotifyAuth } from '../../hooks/useSpotifyAuth.ts';
import { exportPlaylistToSpotify } from '../../services/api.ts';

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
    const { isAuthenticated } = useSpotifyAuth();

    const handleExportToSpotify = async () => {
        if (!isAuthenticated) {
            alert('Musisz być zalogowany do Spotify, aby wyeksportować playlistę');
            return;
        }

        if (playlistItems.length === 0) {
            alert('Playlista jest pusta');
            return;
        }

        setExportingToSpotify(true);
        setExportSuccess(false);

        try {
            const response = await exportPlaylistToSpotify({
                name: playlistName || 'Moja Playlista',
                description: 'Playlista wygenerowana przez Promptify',
                track_ids: playlistItems.map(item => item.trackId),
                public: false,
            });

            setPlaylistUrl(response.playlist_url);
            setExportSuccess(true);

            // Pokaż sukces przez 5 sekund
            setTimeout(() => {
                setExportSuccess(false);
                setPlaylistUrl(null);
            }, 5000);
        } catch (error) {
            console.error('Błąd podczas eksportowania do Spotify:', error);
            alert(`Błąd podczas eksportu: ${error instanceof Error ? error.message : 'Nieznany błąd'}`);
        } finally {
            setExportingToSpotify(false);
        }
    };

    if (exportSuccess && playlistUrl) {
        return (
            <div className="mt-6 flex flex-col items-center gap-3">
                <div className="flex items-center gap-2 text-green-400">
                    <CheckCircle size={24} />
                    <span className="text-lg font-medium">Playlista utworzona!</span>
                </div>
                <a
                    href={playlistUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 px-6 py-3 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
                >
                    <ExternalLink size={20} />
                    Otwórz w Spotify
                </a>
            </div>
        );
    }

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

