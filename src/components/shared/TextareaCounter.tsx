import React from 'react';
import type { CounterMode, CounterState } from '../../types';

export interface TextareaCounterProps {
  currentLength: number;
  maxLength: number;
  mode?: CounterMode;
  warnAt?: number;
  criticalAt?: number;
  className?: string;
}

const TextareaCounter: React.FC<TextareaCounterProps> = ({
  currentLength,
  maxLength,
  mode = 'usedLimit',
  warnAt = 0.9,
  criticalAt = 1.0,
  className = '',
}) => {
  const ratio = maxLength > 0 ? currentLength / maxLength : 0;
  const state: CounterState = ratio >= criticalAt
    ? 'critical'
    : ratio >= warnAt
      ? 'warn'
      : 'normal';

  const text = mode === 'remaining'
    ? `${Math.max(0, maxLength - currentLength)}`
    : `${currentLength}/${maxLength}`;

  return (
    <div
      className={`app-textarea__counter ${state === 'warn' ? 'app-textarea__counter--warn' : ''} ${state === 'critical' ? 'app-textarea__counter--critical' : ''} ${className}`.trim()}
      aria-live="polite"
      aria-atomic="true"
      role="status"
    >
      {text}
    </div>
  );
};

export default TextareaCounter;
