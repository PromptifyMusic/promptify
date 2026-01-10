import { createContext, useContext, useState, useCallback, useRef, useEffect, type ReactNode } from 'react';
import type { PlaylistItem } from '../types';

interface PlaylistState {
  items: PlaylistItem[];
  isExpanded: boolean;
  isLoading: boolean;
  name: string;
  originalPrompt: string;
  initialQuantity: number;
}

interface PlaylistContextType {
  items: PlaylistItem[];
  isExpanded: boolean;
  isLoading: boolean;
  name: string;
  originalPrompt: string;
  initialQuantity: number;

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
  reset: () => void;

  deletingItems: ReadonlySet<string>;
  markAsDeleting: (id: string) => void;
  unmarkAsDeleting: (id: string) => void;
}

const PlaylistContext = createContext<PlaylistContextType | undefined>(undefined);

interface PlaylistProviderProps {
  children: ReactNode;
}

const initialState: PlaylistState = {
  items: [],
  isExpanded: false,
  isLoading: false,
  name: 'Playlista',
  originalPrompt: '',
  initialQuantity: 0,
};

export function PlaylistProvider({ children }: PlaylistProviderProps) {
  // Pojedyncze useState dla każdego pola stanu
  const [items, setItems] = useState<PlaylistItem[]>(initialState.items);
  const [isExpanded, setIsExpanded] = useState(initialState.isExpanded);
  const [isLoading, setIsLoading] = useState(initialState.isLoading);
  const [name, setName] = useState(initialState.name);
  const [originalPrompt, setOriginalPrompt] = useState(initialState.originalPrompt);
  const [initialQuantity, setInitialQuantity] = useState(initialState.initialQuantity);

  const [deletingItems, setDeletingItemsState] = useState<Set<string>>(new Set());
  const deleteTimeoutsRef = useRef<Map<string, number>>(new Map());

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
    setDeletingItemsState(new Set());
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

