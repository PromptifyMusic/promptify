import React, { useState } from 'react';
import Textarea from './Textarea';
import TextareaCounter from './TextareaCounter';

export interface PromptTextareaProps {
  value?: string;
  defaultValue?: string;
  onChange?: (value: string) => void;
  placeholder?: string;
  width?: number | string;
  maxHeight?: number;
  minHeight?: number;
  rows?: number;
  className?: string;
  disabled?: boolean;
  name?: string;
  id?: string;
  ariaLabel?: string;
  maxLength?: number;
  counterMode?: 'usedLimit' | 'remaining';
  warnAt?: number;
  criticalAt?: number;
}

const PromptTextarea: React.FC<PromptTextareaProps> = ({
  value,
  defaultValue = '',
  onChange,
  placeholder,
  width,
  maxHeight = 200,
  minHeight,
  rows = 1,
  className = '',
  disabled = false,
  name,
  id,
  ariaLabel,
  maxLength,
  counterMode = 'usedLimit',
  warnAt = 0.9,
  criticalAt = 1.0,
}) => {
  const isControlled = typeof value === 'string';
  const [mirrorValue, setMirrorValue] = useState<string>(defaultValue);

  const showCounter = typeof maxLength === 'number' && maxLength > 0;
  const currentText = isControlled ? (value as string) : mirrorValue;
  const currentLen = (currentText ?? '').length;

  const handleChange = (next: string) => {
    if (!isControlled) setMirrorValue(next);
    onChange?.(next);
  };

  const textareaClassName = `${className} ${showCounter ? 'app-textarea--with-counter' : ''}`.trim();

  return (
    <div className="app-textarea-wrapper" style={{ width: typeof width === 'number' ? `${width}px` : width }}>
      <Textarea
        value={value}
        defaultValue={defaultValue}
        onChange={handleChange}
        placeholder={placeholder}
        width="100%"
        maxHeight={maxHeight}
        minHeight={minHeight}
        rows={rows}
        className={textareaClassName}
        disabled={disabled}
        name={name}
        id={id}
        ariaLabel={ariaLabel}
        maxLength={maxLength}
      />
      {showCounter && (
        <TextareaCounter
          currentLength={currentLen}
          maxLength={maxLength!}
          mode={counterMode}
          warnAt={warnAt}
          criticalAt={criticalAt}
        />
      )}
    </div>
  );
};

export default PromptTextarea;
