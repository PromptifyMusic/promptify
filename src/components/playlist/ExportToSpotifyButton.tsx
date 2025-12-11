import { memo, useState } from 'react';
import { Music2 } from 'lucide-react';
import ActionButton from '../shared/ActionButton.tsx';
import { PlaylistItem } from './PlaylistSection.tsx';

interface ExportToSpotifyButtonProps {
    playlistName?: string;
    playlistItems: PlaylistItem[];
}

const ExportToSpotifyButton = memo(({
    playlistName,
    playlistItems,
}: ExportToSpotifyButtonProps) => {
    const [exportingToSpotify, setExportingToSpotify] = useState(false);

    const handleExportToSpotify = async () => {
        setExportingToSpotify(true);
        try {
            // Mock API call - symulacja eksportu do Spotify
            await new Promise((resolve) => setTimeout(resolve, 2000));
            console.log('Eksportowanie playlisty do Spotify:', {
                name: playlistName,
                tracks: playlistItems,
            });
            // TODO: Implementacja faktycznego eksportu do Spotify
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
                disabled={exportingToSpotify}
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

