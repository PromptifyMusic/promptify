import { Pencil } from 'lucide-react';
import { useState, useEffect, useRef } from 'react';

interface EditableTitleProps {
  value: string;
  onChange?: (value: string) => void;
  placeholder?: string;
  maxLength?: number;
  className?: string;
  inputClassName?: string;
  iconClassName?: string;
  underlineClassName?: string;
  ariaLabel?: string;
}

const EditableTitle = ({
  value,
  onChange,
  placeholder = 'Wprowadź tytuł',
  maxLength = 50,
  className = '',
  inputClassName = '',
  iconClassName = '',
  underlineClassName = '',
  ariaLabel = 'Edytowalny tytuł',
}: EditableTitleProps) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editedValue, setEditedValue] = useState(value);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setEditedValue(value);
  }, [value]);

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);

  const handleClick = () => {
    setIsEditing(true);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setEditedValue(e.target.value);
  };

  const handleBlur = () => {
    const trimmedValue = editedValue.trim();
    if (trimmedValue && trimmedValue !== value) {
      onChange?.(trimmedValue);
    } else if (!trimmedValue) {
      setEditedValue(value);
    }
    setIsEditing(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      inputRef.current?.blur();
    } else if (e.key === 'Escape') {
      setEditedValue(value);
      setIsEditing(false);
    }
  };

  if (isEditing) {
    return (
      <input
        ref={inputRef}
        type="text"
        value={editedValue}
        onChange={handleChange}
        onBlur={handleBlur}
        onKeyDown={handleKeyDown}
        className={`text-white text-lg font-semibold bg-white/10 border border-white/30 rounded px-1 py-0.5 outline-none focus:border-white/50 transition-colors duration-200 min-w-0 ${inputClassName}`.trim()}
        maxLength={maxLength}
        placeholder={placeholder}
        aria-label={ariaLabel}
        style={{ width: `${Math.max(editedValue.length * 0.6, 8)}em` }}
      />
    );
  }

  return (
    <div
      className={`group flex items-center gap-2 cursor-pointer ${className}`.trim()}
      onClick={handleClick}
      title="Kliknij, aby edytować nazwę"
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          handleClick();
        }
      }}
      aria-label={ariaLabel}
    >
      <h3 className="text-white text-lg font-semibold relative group-hover:text-white/90 transition-colors duration-200">
        {value}
        <span className={`absolute bottom-0 left-0 w-full h-[2px] bg-white/30 group-hover:bg-white/50 transition-colors duration-200 ${underlineClassName}`.trim()} />
      </h3>
      <Pencil className={`w-4 h-4 text-white/50 group-hover:text-white/80 transition-colors duration-200 ${iconClassName}`.trim()} />
    </div>
  );
};

export default EditableTitle;

