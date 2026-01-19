import { createContext, useContext, useState, useCallback, useRef, useEffect, type ReactNode } from 'react';
import type { PlaylistItem } from '../types';

interface PlaylistState {
  items: PlaylistItem[];
  isExpanded: boolean;
  isLoading: boolean;
  name: string;
  originalPrompt: string;
  initialQuantity: number;
  shouldClearPrompt: boolean;
}

interface PlaylistContextType {
  items: PlaylistItem[];
  isExpanded: boolean;
  isLoading: boolean;
  name: string;
  originalPrompt: string;
  initialQuantity: number;
  shouldClearPrompt: boolean;

  setItems: (items: PlaylistItem[]) => void;
  addItem: (item: PlaylistItem) => void;
  updateItem: (id: string, updates: Partial<PlaylistItem>) => void;
  deleteItem: (id: string) => void;
  reorderItems: (items: PlaylistItem[]) => void;
  setIsExpanded: (isExpanded: boolean) => void;
  setIsLoading: (isLoading: boolean) => void;
  setName: (name: string) => void;
  setOriginalPrompt: (prompt: string) => void;
  setInitialQuantity: (quantity: number) => void;
  setShouldClearPrompt: (shouldClear: boolean) => void;
  reset: () => void;

  deletingItems: ReadonlySet<string>;
  markAsDeleting: (id: string) => void;
  unmarkAsDeleting: (id: string) => void;
}

const PlaylistContext = createContext<PlaylistContextType | undefined>(undefined);

interface PlaylistProviderProps {
  children: ReactNode;
}

const STORAGE_KEY = 'promptify_playlist_state';

export const DEFAULT_PLAYLIST_NAME = 'Playlista promptify';

export const initialState: PlaylistState = {
  items: [],
  isExpanded: false,
  isLoading: false,
  name: DEFAULT_PLAYLIST_NAME,
  originalPrompt: '',
  initialQuantity: 0,
  shouldClearPrompt: false,
};

const loadStateFromStorage = (): Partial<PlaylistState> | null => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      return JSON.parse(stored);
    }
  } catch (error) {
    console.error('Error loading playlist state from localStorage:', error);
  }
  return null;
};

const saveStateToStorage = (state: PlaylistState) => {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch (error) {
    console.error('Error saving playlist state to localStorage:', error);
  }
};

const clearStateFromStorage = () => {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch (error) {
    console.error('Error clearing playlist state from localStorage:', error);
  }
};

export function PlaylistProvider({ children }: PlaylistProviderProps) {
  const getInitialState = (): PlaylistState => {
    const stored = loadStateFromStorage();
    return stored ? { ...initialState, ...stored } : initialState;
  };

  const initialStateValue = getInitialState();

  const [items, setItems] = useState<PlaylistItem[]>(initialStateValue.items);
  const [isExpanded, setIsExpanded] = useState(initialStateValue.isExpanded);
  const [isLoading, setIsLoading] = useState(initialState.isLoading);
  const [name, setName] = useState(initialStateValue.name);
  const [originalPrompt, setOriginalPrompt] = useState(initialStateValue.originalPrompt);
  const [initialQuantity, setInitialQuantity] = useState(initialStateValue.initialQuantity);
  const [shouldClearPrompt, setShouldClearPrompt] = useState(initialState.shouldClearPrompt);

  const [deletingItems, setDeletingItemsState] = useState<Set<string>>(new Set());
  const deleteTimeoutsRef = useRef<Map<string, number>>(new Map());

  useEffect(() => {
    const state: PlaylistState = {
      items,
      isExpanded,
      isLoading: false,
      name,
      originalPrompt,
      initialQuantity,
      shouldClearPrompt: false,
    };
    saveStateToStorage(state);
  }, [items, isExpanded, name, originalPrompt, initialQuantity]);

  useEffect(() => {
    return () => {
      deleteTimeoutsRef.current.forEach((timeoutId) => {
        clearTimeout(timeoutId);
      });
      deleteTimeoutsRef.current.clear();
    };
  }, []);

  const addItem = useCallback((item: PlaylistItem) => {
    setItems((prev) => [...prev, item]);
  }, []);

  const updateItem = useCallback((id: string, updates: Partial<PlaylistItem>) => {
    setItems((prev) =>
      prev.map((item) => (item.id === id ? { ...item, ...updates } : item))
    );
  }, []);

  const deleteItem = useCallback(
    (id: string) => {
      if (deletingItems.has(id)) {
        return;
      }

      setDeletingItemsState((prev) => new Set(prev).add(id));

      const timeoutId = setTimeout(() => {
        setItems((prev) => prev.filter((item) => item.id !== id));
        setDeletingItemsState((prev) => {
          const newSet = new Set(prev);
          newSet.delete(id);
          return newSet;
        });
        deleteTimeoutsRef.current.delete(id);
      }, 300);

      deleteTimeoutsRef.current.set(id, timeoutId);
    },
    [deletingItems]
  );

  const reorderItems = useCallback((reorderedItems: PlaylistItem[]) => {
    setItems(reorderedItems);
  }, []);

  const reset = useCallback(() => {
    setItems(initialState.items);
    setIsExpanded(initialState.isExpanded);
    setIsLoading(initialState.isLoading);
    setName(initialState.name);
    setOriginalPrompt(initialState.originalPrompt);
    setInitialQuantity(initialState.initialQuantity);
    setShouldClearPrompt(initialState.shouldClearPrompt);
    setDeletingItemsState(new Set());
    clearStateFromStorage();
  }, []);

  const markAsDeleting = useCallback((id: string) => {
    setDeletingItemsState((prev) => new Set(prev).add(id));
  }, []);

  const unmarkAsDeleting = useCallback((id: string) => {
    setDeletingItemsState((prev) => {
      const newSet = new Set(prev);
      newSet.delete(id);
      return newSet;
    });
  }, []);

  const value: PlaylistContextType = {
    items,
    isExpanded,
    isLoading,
    name,
    originalPrompt,
    initialQuantity,
    shouldClearPrompt,

    setItems,
    addItem,
    updateItem,
    deleteItem,
    reorderItems,
    setIsExpanded,
    setIsLoading,
    setName,
    setOriginalPrompt,
    setInitialQuantity,
    setShouldClearPrompt,
    reset,

    deletingItems,
    markAsDeleting,
    unmarkAsDeleting,
  };

  return (
    <PlaylistContext.Provider value={value}>
      {children}
    </PlaylistContext.Provider>
  );
}

export function usePlaylistContext(): PlaylistContextType {
  const context = useContext(PlaylistContext);

  if (context === undefined) {
    throw new Error('usePlaylistContext must be used within PlaylistProvider');
  }

  return context;
}
