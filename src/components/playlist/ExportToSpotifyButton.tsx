import { memo, useState } from 'react';
import { Music2 } from 'lucide-react';
import ActionButton from '../shared/ActionButton.tsx';
import { PlaylistItem } from './PlaylistSection.tsx';
import { useSpotifyAuth } from '../../hooks/useSpotifyAuth.ts';

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
            console.warn('Użytkownik nie jest zalogowany do Spotify');
            return;
        }

        setExportingToSpotify(true);
        try {
            // Mock API call - symulacja eksportu do Spotify
            await new Promise((resolve) => setTimeout(resolve, 2000));

            // TODO: Implementacja faktycznego eksportu do Spotify
            // Używamy trackId - to właściwy ID utworu w Spotify
            console.log('Eksportowanie playlisty do Spotify:', {
                name: playlistName,
                trackIds: playlistItems.map(item => item.trackId),
            });
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
                Eksportuj do Spotify
            </ActionButton>
        </div>
    );
});

ExportToSpotifyButton.displayName = 'ExportToSpotifyButton';

export default ExportToSpotifyButton;

