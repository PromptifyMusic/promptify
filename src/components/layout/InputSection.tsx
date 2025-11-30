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

function InputSection({ isPlaylistExpanded, onCreatePlaylist, isLoading = false }: InputSectionProps) {
    const [prompt, setPrompt] = useState<string>("");
    const [quantity, setQuantity] = useState<number>(1);

    const handleCreateClick = () => {
        onCreatePlaylist(prompt, quantity);
    };

    return (
        <div className={`logo-container flex flex-col items-center justify-center gap-6 ${isPlaylistExpanded ? 'fade-out' : 'fade-in'}`}>
            <Logo className="mb-16" />

            <div>
                <PromptTextarea
                    value={prompt}
                    onChange={setPrompt}
                    maxLength={250}
                    placeholder="Wprowadź prompt do utworzenia playlisty"
                    width={600}
                />
            </div>
            <div className="flex flex-col items-center gap-2 mb-5">
                <QuantityInput
                    min={1}
                    max={10}
                    defaultValue={1}
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

