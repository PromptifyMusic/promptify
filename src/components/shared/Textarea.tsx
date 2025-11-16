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

    const showCounter = typeof maxLength === 'number';
    const currentLen = (currentValue ?? '').length;
    const ratio = showCounter && maxLength ? currentLen / maxLength : 0;
    const counterState: 'normal' | 'warn' | 'critical' = !showCounter
        ? 'normal'
        : ratio >= criticalAt
            ? 'critical'
            : ratio >= warnAt
                ? 'warn'
                : 'normal';
    const counterText = showCounter
        ? (counterMode === 'remaining'
            ? `${Math.max(0, (maxLength ?? 0) - currentLen)}`
            : `${currentLen}/${maxLength}`)
        : '';

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
                <div
                    className={`app-textarea__counter ${counterState === 'warn' ? 'app-textarea__counter--warn' : ''} ${counterState === 'critical' ? 'app-textarea__counter--critical' : ''}`.trim()}
                    aria-hidden="true"
                    aria-live="polite"
                    role="status"
                >
                    {counterText}
                </div>
            )}
        </div>
    );
};

export default Textarea;
