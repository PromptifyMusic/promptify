import { ChevronUp } from 'lucide-react';
import { useState, useEffect } from 'react';
import '../../styles/ExpandablePlaylistBox.css';
import EditableTitle from '../shared/EditableTitle';

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

  const handleCollapse = () => {
    if (onCollapse) {
      onCollapse();
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
            <EditableTitle
              value={playlistName}
              onChange={onPlaylistNameChange}
              placeholder="Nazwa playlisty"
              maxLength={50}
              ariaLabel="Edytuj nazwę playlisty"
            />
            <button
              onClick={handleCollapse}
              className="flex items-center gap-2 px-4 py-2 hover:bg-white/10 rounded-full transition-colors duration-200"
              aria-label="Zwiń playlistę"
            >
              <span className="text-white text-sm font-medium">Powrót</span>
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
