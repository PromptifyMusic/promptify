import { useState } from "react";
import QuantityInput from "../prompt/QuantityInput.tsx";
import PromptTextarea from "../prompt/PromptTextarea.tsx";
import ActionButton from "../shared/ActionButton.tsx";
import Logo from "./Logo.tsx";

interface InputSectionProps {
    isPlaylistExpanded: boolean;
    onCreatePlaylist: (prompt: string, quantity: number) => void;
    isLoading?: boolean;
}

const DEFAULT_QUANTITY = 15;

function InputSection({ isPlaylistExpanded, onCreatePlaylist, isLoading = false }: InputSectionProps) {
    const [prompt, setPrompt] = useState<string>("");
    const [quantity, setQuantity] = useState<number>(DEFAULT_QUANTITY);
    const [hasError, setHasError] = useState<boolean>(false);

    const handleCreateClick = () => {
        const trimmedPrompt = prompt.trim();
        if (!trimmedPrompt) {
            setHasError(true);
            return;
        }
        setHasError(false);
        onCreatePlaylist(trimmedPrompt, quantity);
    };

    const handlePromptChange = (value: string) => {
        setPrompt(value);
        if (hasError && value.trim()) {
            setHasError(false);
        }
    };

    return (
        <div className={`logo-container flex flex-col items-center justify-center gap-6 ${isPlaylistExpanded ? 'fade-out' : 'fade-in'}`}>
            <Logo className="mb-16" />

            <div>
                <PromptTextarea
                    value={prompt}
                    onChange={handlePromptChange}
                    maxLength={250}
                    placeholder="Wprowadź prompt do utworzenia playlisty"
                    width={600}
                    hasError={hasError}
                    errorMessage="Proszę wprowadzić prompt"
                />
            </div>
            <div className="flex flex-col items-center gap-2 mb-5">
                <QuantityInput
                    min={1}
                    max={50}
                    defaultValue={DEFAULT_QUANTITY}
                    onChange={setQuantity}
                />
                <span className="text-white/50 text-sm">
                    Liczba utworów w playliście
                </span>
            </div>
            <ActionButton className='bg-white rounded-md' onClick={handleCreateClick} loading={isLoading}>
                Utwórz playlistę
            </ActionButton>
        </div>
    );
}

export default InputSection;

