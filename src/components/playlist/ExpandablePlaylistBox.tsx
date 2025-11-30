import { ChevronUp, Pencil } from 'lucide-react';
import { useState, useEffect, useRef } from 'react';
import '../../styles/ExpandablePlaylistBox.css';

const ANIMATION_DURATION = 500;

interface ExpandablePlaylistBoxProps {
  maxWidth?: string;
  minWidth?: string;
  maxHeight?: string;
  children?: React.ReactNode;
  isExpanded?: boolean;
  onCollapse?: () => void;
  playlistName?: string;
  onPlaylistNameChange?: (name: string) => void;
}

const ExpandablePlaylistBox = ({
  maxWidth = '800px',
  minWidth = '400px',
  maxHeight = '600px',
  children,
  isExpanded = false,
  onCollapse,
  playlistName = 'Playlista',
  onPlaylistNameChange,
}: ExpandablePlaylistBoxProps) => {
  const [isAnimationComplete, setIsAnimationComplete] = useState(false);
  const [isEditingName, setIsEditingName] = useState(false);
  const [editedName, setEditedName] = useState(playlistName);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isExpanded) {
      const timer = setTimeout(() => {
        setIsAnimationComplete(true);
      }, ANIMATION_DURATION);
      return () => clearTimeout(timer);
    } else {
      setIsAnimationComplete(false);
    }
  }, [isExpanded]);

  useEffect(() => {
    setEditedName(playlistName);
  }, [playlistName]);

  useEffect(() => {
    if (isEditingName && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditingName]);

  const handleCollapse = () => {
    if (onCollapse) {
      onCollapse();
    }
  };

  const handleNameClick = () => {
    setIsEditingName(true);
  };

  const handleNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setEditedName(e.target.value);
  };

  const handleNameBlur = () => {
    const trimmedName = editedName.trim();
    if (trimmedName && trimmedName !== playlistName) {
      onPlaylistNameChange?.(trimmedName);
    } else if (!trimmedName) {
      setEditedName(playlistName);
    }
    setIsEditingName(false);
  };

  const handleNameKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      inputRef.current?.blur();
    } else if (e.key === 'Escape') {
      setEditedName(playlistName);
      setIsEditingName(false);
    }
  };

  return (
    <div
      className="relative w-full transition-all duration-500 ease-in-out overflow-hidden"
      style={{
        height: isExpanded ? maxHeight : '4px',
      }}
    >
      <div
        className={`absolute inset-0 transition-opacity duration-300 ${
          isExpanded ? 'opacity-0 pointer-events-none' : 'opacity-100'
        }`}
      >
        <div className="w-full h-full bg-gradient-to-r from-transparent via-white/30 to-transparent rounded-full" />
      </div>

      <div
        className={`absolute inset-0 flex items-start justify-center transition-all duration-500 ${
          isExpanded
            ? 'opacity-100 scale-100'
            : 'opacity-0 scale-95 pointer-events-none'
        }`}
      >
        <div
          className="h-full backdrop-blur-md bg-white/10 border border-white/20 rounded-lg shadow-2xl overflow-hidden flex flex-col"
          style={{ maxHeight, maxWidth, minWidth }}
        >
          <div className="flex items-center justify-between p-4 border-b border-white/20">
            {isEditingName ? (
              <input
                ref={inputRef}
                type="text"
                value={editedName}
                onChange={handleNameChange}
                onBlur={handleNameBlur}
                onKeyDown={handleNameKeyDown}
                className="text-white text-lg font-semibold bg-white/10 border border-white/30 rounded px-1 py-0.5 outline-none focus:border-white/50 transition-colors duration-200 min-w-0"
                maxLength={50}
                style={{ width: `${Math.max(editedName.length * 0.6, 8)}em` }}
              />
            ) : (
              <div
                className="group flex items-center gap-2 cursor-pointer"
                onClick={handleNameClick}
                title="Kliknij, aby edytować nazwę"
              >
                <h3 className="text-white text-lg font-semibold relative group-hover:text-white/90 transition-colors duration-200">
                  {playlistName}
                  <span className="absolute bottom-0 left-0 w-full h-[2px] bg-white/30 group-hover:bg-white/50 transition-colors duration-200" />
                </h3>
                <Pencil className="w-4 h-4 text-white/50 group-hover:text-white/80 transition-colors duration-200" />
              </div>
            )}
            <button
              onClick={handleCollapse}
              className="p-2 hover:bg-white/10 rounded-full transition-colors duration-200"
              aria-label="Zwiń playlistę"
            >
              <ChevronUp className="w-5 h-5 text-white" />
            </button>
          </div>

          <div className={`flex-1 overflow-y-auto p-4 ${
            isAnimationComplete ? 'playlist-scrollbar' : 'playlist-scrollbar-hidden'
          }`}>
            {children || (
              <div className="text-white/70 text-center py-8">
                Brak elementów w playliście
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ExpandablePlaylistBox;

