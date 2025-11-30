import React from 'react';
import { AudioLines } from 'lucide-react';
import '../../styles/Logo.css';

interface LogoProps {
  isHidden?: boolean;
  className?: string;
}

const Logo: React.FC<LogoProps> = ({ className = '' }) => {
  return (
    <div className={`flex items-center gap-3 ${className}`}>
      <AudioLines className="w-10 h-10 text-white/90" strokeWidth={2} />
      <h1 className="text-5xl font-bold text-white/90 tracking-wide">
        promptify
      </h1>
    </div>
  );
};

export default Logo;

