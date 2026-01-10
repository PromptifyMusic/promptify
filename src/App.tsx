import DarkVeil from "./components/layout/animatedBackground/DarkVeil";
import SpotifyAuth from "./components/playlist/SpotifyAuth";
import { PlaylistProvider } from "./context/PlaylistContext";
import { InputSectionContainer } from "./components/containers/InputSectionContainer";
import { PlaylistSectionContainer } from "./components/containers/PlaylistSectionContainer";

function App() {
  return (
    <PlaylistProvider>
      <AppContent />
    </PlaylistProvider>
  );
}

function AppContent() {
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
      <SpotifyAuth />
      <div className="relative z-10 w-full h-full flex flex-col items-center justify-center gap-8 p-8">
        <InputSectionContainer />
        <PlaylistSectionContainer />
      </div>
    </div>
  );
}

export default App;
