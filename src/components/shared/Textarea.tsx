import React, { useState } from 'react';
import '../../styles/Textarea.css';
import { useAutoResizeTextarea } from '../../hooks/useAutoResizeTextarea';
import TextareaCounter from './TextareaCounter';

interface TextareaProps {
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

const Textarea: React.FC<TextareaProps> = ({
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
    const isControlled = value !== undefined;
    const [internalValue, setInternalValue] = useState<string>(defaultValue);

    const currentValue = isControlled ? (value as string) : internalValue;

    const ref = useAutoResizeTextarea({ value: currentValue ?? '', maxHeight, minHeight });

    const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        let next = e.target.value;
        if (typeof maxLength === 'number' && next.length > maxLength) {
            next = next.slice(0, maxLength);
        }
        if (!isControlled) {
            setInternalValue(next);
        }
        onChange?.(next);
    };

    const resolvedWidth = typeof width === 'number' ? `${width}px` : width;

    const textareaStyle: React.CSSProperties = {maxHeight};
    if (minHeight !== undefined) {
        textareaStyle.minHeight = minHeight;
    }

    const showCounter = typeof maxLength === 'number' && maxLength > 0;
    const currentLen = (currentValue ?? '').length;

    return (
        <div className="app-textarea-wrapper" style={{ width: resolvedWidth }}>
            <textarea
                ref={ref}
                id={id}
                name={name}
                className={`app-textarea glass glass--inset ${showCounter ? 'app-textarea--with-counter' : ''} ${className}`.trim()}
                placeholder={placeholder}
                value={currentValue}
                onChange={handleChange}
                disabled={disabled}
                aria-label={ariaLabel || placeholder || 'Textarea'}
                role="textbox"
                rows={rows}
                style={textareaStyle}
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

export default Textarea;
