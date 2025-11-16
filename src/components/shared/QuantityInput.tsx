import React, { useState, useEffect } from 'react';
import '../../styles/QuantityInput.css';

interface QuantityInputProps {
  min?: number;
  max?: number;
  defaultValue?: number;
  onChange?: (value: number) => void;
  className?: string;
}

const QuantityInput: React.FC<QuantityInputProps> = ({
  min = 1,
  max = 999,
  defaultValue = 1,
  onChange,
  className = '',
}) => {
  const [value, setValue] = useState<number>(defaultValue);
  const [inputValue, setInputValue] = useState<string>(String(defaultValue));

  useEffect(() => {
    let clampedValue = defaultValue;
    if (defaultValue < min) {
      clampedValue = min;
    } else if (defaultValue > max) {
      clampedValue = max;
    }

    setValue(clampedValue);
    setInputValue(String(clampedValue));

    if (clampedValue !== defaultValue) {
      onChange?.(clampedValue);
    }
  }, [defaultValue, min, max, onChange]);

  const commitValue = (next: number) => {
    const clamped = Math.max(min, Math.min(max, next));
    setValue(clamped);
    setInputValue(String(clamped));
    if (clamped !== value) {
      onChange?.(clamped);
    }
  };

  const handleIncrement = () => {
    commitValue(value + 1);
  };

  const handleDecrement = () => {
    commitValue(value - 1);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value;

    if (raw === '') {
      setInputValue('');
      return;
    }

    if (/^\d*$/.test(raw)) {
      setInputValue(raw);
    }
  };

  const handleBlur = () => {
    const parsed = parseInt(inputValue, 10);
    const next = isNaN(parsed) ? min : parsed;
    commitValue(next);
  };

  return (
    <div className={`quantity-input ${className}`}>
      <button
        onClick={handleDecrement}
        disabled={value <= min}
        className="quantity-input__button quantity-input__button--decrement"
        aria-label="Decrease quantity"
      >
        −
      </button>

      <input
        type="text"
        value={inputValue}
        onChange={handleInputChange}
        onBlur={handleBlur}
        className="quantity-input__field"
        aria-label="Quantity"
        role="spinbutton"
        aria-valuemin={min}
        aria-valuemax={max}
        aria-valuenow={value}
        inputMode="numeric"
        pattern="[0-9]*"
      />

      <button
        onClick={handleIncrement}
        disabled={value >= max}
        className="quantity-input__button quantity-input__button--increment"
        aria-label="Increase quantity"
      >
        +
      </button>
    </div>
  );
};

export default QuantityInput;
