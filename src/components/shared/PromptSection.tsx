import React from 'react';
import PromptTextarea from './PromptTextarea';
import QuantityInput from './QuantityInput';
import ActionButton from './ActionButton';
import '../../styles/PromptSection.css';

interface PromptSectionProps {
  isHidden?: boolean;
  onCreatePlaylist: () => void;
  className?: string;
}

const PromptSection: React.FC<PromptSectionProps> = ({
  isHidden = false,
  onCreatePlaylist,
  className = ''
}) => {
  return (
    <div className={`prompt-section ${isHidden ? 'fade-out' : 'fade-in'} ${className}`}>
      <div className="w-1/3">
        <PromptTextarea
          maxLength={250}
          placeholder="Wprowadź prompt do utworzenia playlisty"
        />
      </div>
      <div className="flex flex-col items-center gap-2">
        <QuantityInput min={1} max={10} defaultValue={1} />
        <span className="text-white/50 text-sm">
          Liczba utworów w playliście
        </span>
      </div>
      <ActionButton className='bg-white rounded-md' onClick={onCreatePlaylist}>
        Utwórz playlistę
      </ActionButton>
    </div>
  );
};

export default PromptSection;

