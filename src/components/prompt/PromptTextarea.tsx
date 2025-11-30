import React, { useState } from 'react';
import Textarea from '../shared/Textarea.tsx';
import TextareaCounter from '../shared/TextareaCounter.tsx';

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
  hasError?: boolean;
  errorMessage?: string;
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
  hasError = false,
  errorMessage,
}) => {
  const isControlled = value !== undefined;
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
        hasError={hasError}
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
      {hasError && errorMessage && (
        <div className="text-red-500 text-xs mt-1 px-1">
          {errorMessage}
        </div>
      )}
    </div>
  );
};

export default PromptTextarea;
