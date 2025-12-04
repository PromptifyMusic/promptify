import React, { useState } from 'react';
import '../../styles/Textarea.css';
import { useAutoResizeTextarea } from '../../hooks/useAutoResizeTextarea';

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
    const style: React.CSSProperties = { maxHeight, width: resolvedWidth };
    if (minHeight !== undefined) style.minHeight = minHeight;

    return (
        <textarea
            ref={ref}
            id={id}
            name={name}
            className={`app-textarea glass glass--inset ${className}`.trim()}
            placeholder={placeholder}
            value={currentValue}
            onChange={handleChange}
            disabled={disabled}
            aria-label={ariaLabel || placeholder || 'Textarea'}
            rows={rows}
            style={style}
            maxLength={maxLength}
        />
    );
};

export default Textarea;
