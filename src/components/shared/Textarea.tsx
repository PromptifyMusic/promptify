import React, {useEffect, useRef, useState} from 'react';
import '../../styles/Textarea.css';

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
    const ref = useRef<HTMLTextAreaElement | null>(null);

    const currentValue = isControlled ? (value as string) : internalValue;

    const baselineRef = useRef<number | null>(null);

    const adjustHeight = () => {
        const el = ref.current;

        if (!el) {
            return;
        }

        el.style.height = 'auto';
        const scrollH = el.scrollHeight;
        const limit = maxHeight;

        if (baselineRef.current === null) {
            baselineRef.current = scrollH;
        }
        const minH = minHeight ?? baselineRef.current ?? scrollH;

        if (scrollH > limit) {
            el.style.height = limit + 'px';
            el.style.overflowY = 'auto';
        } else {
            const target = Math.max(scrollH, minH);
            el.style.height = target + 'px';
            el.style.overflowY = 'hidden';
        }
    };

    useEffect(() => {
        adjustHeight();
    }, [currentValue, maxHeight, minHeight]);

    const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        let next = e.target.value;
        if (maxLength !== undefined && next.length > maxLength) {
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

    const showCounter = true;
    const currentLen = (currentValue ?? '').length;
    const counterText = maxLength !== undefined ? `${currentLen}/${maxLength}` : `${currentLen}`;

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
                <div className="app-textarea__counter" aria-hidden="true">
                    {counterText}
                </div>
            )}
        </div>
    );
};

export default Textarea;
