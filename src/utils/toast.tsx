import toast from 'react-hot-toast';
import { Info, AlertTriangle } from 'lucide-react';

/**
 * Toast notification utility functions
 * Provides centralized error handling and user notifications
 */

export const showToast = {
  /**
   * Show a success toast notification
   */
  success: (message: string) => {
    toast.success(message, {
      duration: 4000,
    });
  },

  /**
   * Show an error toast notification
   */
  error: (message: string, error?: unknown) => {
    // Log error to console for debugging
    if (error) {
      console.error('[Toast Error]:', error);
    }

    toast.error(message, {
      duration: 5000,
    });
  },

  /**
   * Show a loading toast notification
   * Returns the toast id which can be used to dismiss it later
   */
  loading: (message: string) => {
    return toast.loading(message);
  },

  /**
   * Show an info toast notification
   */
  info: (message: string) => {
    toast(message, {
      icon: <Info size={20} />,
      duration: 4000,
    });
  },

  /**
   * Show a warning toast notification
   */
  warning: (message: string) => {
    toast(message, {
      icon: <AlertTriangle size={20} />,
      duration: 4000,
    });
  },

  /**
   * Dismiss a specific toast by id
   */
  dismiss: (toastId?: string) => {
    toast.dismiss(toastId);
  },

  /**
   * Promise-based toast for async operations
   */
  promise: <T,>(
    promise: Promise<T>,
    messages: {
      loading: string;
      success: string;
      error: string;
    }
  ) => {
    return toast.promise(promise, messages);
  },
};

/**
 * Helper function to extract error message from various error types
 */
export const getErrorMessage = (error: unknown): string => {
  if (error instanceof Error) {
    return error.message;
  }
  if (typeof error === 'string') {
    return error;
  }
  return 'Wystąpił nieznany błąd';
};

/**
 * Check if error is a network/connection error
 */
export const isNetworkError = (error: unknown): boolean => {
  if (error instanceof TypeError) {
    const message = error.message.toLowerCase();
    return message.includes('failed to fetch') ||
           message.includes('network') ||
           message.includes('connection');
  }
  return false;
};

/**
 * Handle API errors and show appropriate toast notifications
 * Automatically detects network errors and shows appropriate messages
 */
export const handleApiError = (error: unknown, contextMessage: string) => {
  const errorMessage = getErrorMessage(error);

  console.error(`[API Error] ${contextMessage}:`, error);

  if (isNetworkError(error)) {
    showToast.error('Nie można połączyć się z serwerem.');
  } else {
    showToast.error(contextMessage);
  }

  return errorMessage;
};
