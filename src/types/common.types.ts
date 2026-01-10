/**
 * Common Types
 * Wspólne typy używane w całej aplikacji
 */

/**
 * Generyczny typ dla callbacków
 */
export type VoidCallback = () => void;
export type Callback<T> = (value: T) => void;

/**
 * Typ dla handlera zmiany inputu
 */
export type ChangeHandler<T = string> = (value: T) => void;

/**
 * Wynik walidacji
 */
export interface ValidationResult {
  valid: boolean;
  error?: string;
}

/**
 * Typ dla stanu ładowania/błędu
 */
export interface AsyncState<T> {
  data: T | null;
  isLoading: boolean;
  error: Error | null;
}

/**
 * Poziomy ostrzeżeń dla counter'a
 */
export enum CounterState {
  Normal = 'normal',
  Warn = 'warn',
  Critical = 'critical',
}

/**
 * Tryb wyświetlania licznika
 */
export enum CounterMode {
  UsedLimit = 'usedLimit',
  Remaining = 'remaining',
}

/**
 * Opcje dla auto-resize textarea
 */
export interface AutoResizeOptions {
  value: string;
  maxHeight: number;
  minHeight?: number;
}

