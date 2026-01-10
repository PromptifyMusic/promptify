# Types

Centralny folder z definicjami typów TypeScript dla całej aplikacji.

## 📁 Struktura

### `index.ts`
Główny punkt eksportu wszystkich typów. **Zawsze importuj typy z tego pliku** dla spójności:

```typescript
import type { PlaylistItem, ApiPlaylistTrack, SpotifyUser } from '../types';
```

### `api.types.ts`
Typy związane z komunikacją API z backendem:
- `ApiPlaylistTrack` - reprezentacja utworu z API
- `ExportPlaylistRequest` - request do eksportu playlisty
- `ExportPlaylistResponse` - response po eksporcie
- `GeneratePlaylistRequest` - request do generowania playlisty
- `ReplaceSongRequest` - request do zamiany utworu
- `ApiError` - struktura błędu API
- `TempPlaylistTrack` ⚠️ *deprecated* - legacy type

### `playlist.types.ts`
Typy związane z playlistą w UI aplikacji:
- `PlaylistItem` - reprezentacja utworu w UI
- `PlaylistState` - stan całej playlisty
- `PlaylistAction` - akcje dla reducera playlisty

### `spotify.types.ts`
Typy związane z integracją Spotify:
- `SpotifyUser` - reprezentacja użytkownika Spotify
- `SpotifyImage` - obraz z API Spotify
- `SpotifyAuthStatus` - status autoryzacji
- `UseSpotifyAuthReturn` - return type dla hooka useSpotifyAuth

### `common.types.ts`
Wspólne typy używane w różnych częściach aplikacji:
- `VoidCallback`, `Callback<T>` - typy dla callbacków
- `ChangeHandler<T>` - handlery zmian
- `ValidationResult` - wynik walidacji
- `AsyncState<T>` - stan asynchroniczny
- `CounterState`, `CounterMode` - typy dla liczników
- `AutoResizeOptions` - opcje dla auto-resize textarea

## 📋 Zasady

1. **Zawsze importuj z `index.ts`**:
   ```typescript
   ✅ import type { PlaylistItem } from '../types';
   ❌ import type { PlaylistItem } from '../types/playlist.types';
   ```

2. **Używaj `type` imports dla type-only imports**:
   ```typescript
   ✅ import type { PlaylistItem } from '../types';
   ❌ import { PlaylistItem } from '../types';
   ```

3. **Dokumentuj nowe typy z JSDoc**:
   ```typescript
   /**
    * Reprezentacja użytkownika w systemie
    */
   export interface User {
     id: string;
     name: string;
   }
   ```

4. **Nazewnictwo**:
   - Interfejsy: `PascalCase` (np. `PlaylistItem`)
   - Type aliases: `PascalCase` (np. `PlaylistAction`)
   - Prefiksy:
     - `Api*` - typy z/do API
     - `Use*Return` - return types dla hooków
     - Brak prefiksu dla typów UI

5. **Gdzie umieszczać typy**:
   - `api.types.ts` - request/response, struktury danych z backendu
   - `playlist.types.ts` - logika biznesowa playlisty
   - `spotify.types.ts` - wszystko związane ze Spotify API
   - `common.types.ts` - typy używane w wielu miejscach
   - Props komponentów - mogą zostać w plikach komponentów jeśli są używane tylko lokalnie

## 🔄 Migracja z poprzedniego kodu

Przed refaktorem typy były rozrzucone po różnych plikach:
- `PlaylistItem` był w `components/playlist/PlaylistSection.tsx`
- `SpotifyUser` był w `hooks/useSpotifyAuth.ts`
- `ExportPlaylistRequest` był w `services/api.ts`

Teraz wszystkie typy są w dedykowanych plikach i można je łatwo znaleźć i reużyć.

## 📚 Przykłady użycia

### W komponencie:
```typescript
import type { PlaylistItem } from '../../types';

interface PlaylistSectionProps {
  items: PlaylistItem[];
  onItemClick: (item: PlaylistItem) => void;
}
```

### W hooku:
```typescript
import type { PlaylistItem, ApiPlaylistTrack } from '../types';

export function usePlaylistOperations() {
  const [items, setItems] = useState<PlaylistItem[]>([]);
  // ...
}
```

### W service:
```typescript
import type { ApiPlaylistTrack, ExportPlaylistRequest } from '../types';

export async function generatePlaylist(): Promise<ApiPlaylistTrack[]> {
  // ...
}
```

## 🚀 Rozwój

Przy dodawaniu nowych typów:

1. Określ do której kategorii należy typ
2. Dodaj go do odpowiedniego pliku (`*.types.ts`)
3. Wyeksportuj z `index.ts`
4. Dodaj dokumentację JSDoc
5. Zaktualizuj ten README jeśli dodajesz nową kategorię

---

*Utworzono: 2026-01-10*
*Zgodne z: TypeScript 5.9.3, React 19.1.1*

