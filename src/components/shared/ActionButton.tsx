import React from 'react';
import '../../styles/ActionButton.css'

export interface ActionButtonProps {
  onClick?: () => void;
  disabled?: boolean;
  loading?: boolean;
  children?: React.ReactNode;
  className?: string;
  type?: 'button' | 'submit' | 'reset';
}

const ActionButton = ({
    onClick,
    disabled = false,
    loading = false,
    children,
    className = '',
    type = 'button',
}: ActionButtonProps) => {
    return (
        <button
            type={type}
            className={`action-button ${className} ${loading ? 'action-button--loading' : ''}`.trim()}
            onClick={onClick}
            disabled={disabled || loading}
        >
            <span className={`action-button__content ${loading ? 'action-button__content--hidden' : ''}`}>
                {children}
            </span>
            {loading && <span className="action-button__spinner" aria-label="Loading"></span>}
        </button>
    );
}

export default ActionButton;
