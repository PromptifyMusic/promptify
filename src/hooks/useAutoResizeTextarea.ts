import { useEffect, useRef } from 'react';
import type { AutoResizeOptions } from '../types';

export function useAutoResizeTextarea({ value, maxHeight, minHeight }: AutoResizeOptions) {
  const ref = useRef<HTMLTextAreaElement | null>(null);
  const baselineRef = useRef<number | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    el.style.height = 'auto';
    const scrollH = el.scrollHeight;
    const limit = maxHeight;

    if (baselineRef.current === null) {
      baselineRef.current = scrollH;
    }
    const minH = (typeof minHeight === 'number' ? minHeight : baselineRef.current ?? scrollH);

    if (scrollH > limit) {
      el.style.height = `${limit}px`;
      el.style.overflowY = 'auto';
    } else {
      const target = Math.max(scrollH, minH);
      el.style.height = `${target}px`;
      el.style.overflowY = 'hidden';
    }
  }, [value, maxHeight, minHeight]);

  return ref;
}

