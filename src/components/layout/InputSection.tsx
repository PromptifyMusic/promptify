import QuantityInput from "../shared/QuantityInput";
import PromptTextarea from "../shared/PromptTextarea";
import ActionButton from "../shared/ActionButton";
import Logo from "../shared/Logo";

interface InputSectionProps {
    isPlaylistExpanded: boolean;
    onCreatePlaylist: () => void;
}

function InputSection({ isPlaylistExpanded, onCreatePlaylist }: InputSectionProps) {
    return (
        <div className={`logo-container flex flex-col items-center justify-center gap-6 ${isPlaylistExpanded ? 'fade-out' : 'fade-in'}`}>
            <Logo className="mb-15" />

            <div>
                <PromptTextarea
                    maxLength={250}
                    placeholder="Wprowadź prompt do utworzenia playlisty"
                    width={600}
                />
            </div>
            <div className="flex flex-col items-center gap-2">
                <QuantityInput min={1} max={10} defaultValue={1} />
                <span className="text-white/50 text-sm">
                    Liczba utworów w playliście
                </span>
            </div>
            <ActionButton className='bg-white rounded-md' onClick={onCreatePlaylist}>
                Utwórz playlistę
            </ActionButton>
        </div>
    );
}

export default InputSection;

