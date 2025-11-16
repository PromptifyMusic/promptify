import './App.css'
// import DarkVeil from "./components/layout/background/DarkVeil.tsx";
import DarkVeil from "./components/DarkVeil/DarkVeil.tsx";

function App() {
    return (
        <div className="w-full h-full">
            <DarkVeil 
                hueShift={180}
                speed={0.25}
                warpAmount={1}
                resolutionScale={1}
            />
        </div>
    )
}

export default App
