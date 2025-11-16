import DarkVeil from "./components/layout/animatedBackground/DarkVeil.tsx";
import QuantityInput from "./components/shared/QuantityInput";
import PromptTextarea from "./components/shared/PromptTextarea.tsx";

function App() {
    return (
        <div className="relative w-full h-screen overflow-hidden">
            <div className="absolute inset-0 -z-10">
                <DarkVeil 
                    hueShift={180}
                    speed={0.25}
                    warpAmount={1}
                    resolutionScale={1}
                />
            </div>

            <div className="relative z-10 w-full h-full flex flex-col items-center justify-center gap-4">
                <div className="w-1/3">
                    <PromptTextarea
                        maxLength={250}
                        placeholder="Wprowadź prompt do utworzenia playlisty"
                    />
                </div>
                <QuantityInput min={1} max={10} defaultValue={1} />
            </div>
        </div>
    )
}

export default App
