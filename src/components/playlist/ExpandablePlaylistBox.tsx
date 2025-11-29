import { ChevronUp } from 'lucide-react';
import '../../styles/ExpandablePlaylistBox.css';

interface ExpandablePlaylistBoxProps {
  maxWidth?: string;
  maxHeight?: string;
  children?: React.ReactNode;
  isExpanded?: boolean;
  onCollapse?: () => void;
}

const ExpandablePlaylistBox = ({
  maxWidth = '800px',
  maxHeight = '600px',
  children,
  isExpanded = false,
  onCollapse,
}: ExpandablePlaylistBoxProps) => {
  const handleCollapse = () => {
    if (onCollapse) {
      onCollapse();
    }
  };

  return (
    <div
      className="relative transition-all duration-500 ease-in-out overflow-hidden"
      style={{
        maxWidth: isExpanded ? maxWidth : '100%',
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
        className={`absolute inset-0 transition-all duration-500 ${
          isExpanded
            ? 'opacity-100 scale-100'
            : 'opacity-0 scale-95 pointer-events-none'
        }`}
      >
        <div
          className="w-full h-full backdrop-blur-md bg-white/10 border border-white/20 rounded-lg shadow-2xl overflow-hidden flex flex-col"
          style={{ maxHeight }}
        >
          <div className="flex items-center justify-between p-4 border-b border-white/20">
            <h3 className="text-white text-lg font-semibold">Playlista</h3>
            <button
              onClick={handleCollapse}
              className="p-2 hover:bg-white/10 rounded-full transition-colors duration-200"
              aria-label="Zwiń playlistę"
            >
              <ChevronUp className="w-5 h-5 text-white" />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-4 scrollbar-hide">
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

