import DarkVeil from "./components/layout/animatedBackground/DarkVeil.tsx";
import QuantityInput from "./components/shared/QuantityInput";

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
                <QuantityInput min={1} max={10} defaultValue={1} />
            </div>
        </div>
    )
}

export default App
