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

  useEffect(() => {
    if (defaultValue < min) {
      setValue(min);
    } else if (defaultValue > max) {
      setValue(max);
    } else {
      setValue(defaultValue);
    }
  }, [defaultValue, min, max]);

  const handleIncrement = () => {
    const newValue = Math.min(value + 1, max);
    setValue(newValue);
    onChange?.(newValue);
  };

  const handleDecrement = () => {
    const newValue = Math.max(value - 1, min);
    setValue(newValue);
    onChange?.(newValue);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const inputValue = e.target.value;

    if (inputValue === '') {
      setValue(min);
      onChange?.(min);
      return;
    }

    const numericValue = parseInt(inputValue, 10);
    
    if (!isNaN(numericValue)) {
      const clampedValue = Math.max(min, Math.min(max, numericValue));
      setValue(clampedValue);
      onChange?.(clampedValue);
    }
  };

  const handleBlur = () => {
    if (value < min) {
      setValue(min);
      onChange?.(min);
    } else if (value > max) {
      setValue(max);
      onChange?.(max);
    }
  };

  return (
    <div className={`quantity-input ${className}`}>
      <button
        onClick={handleDecrement}
        disabled={value <= min}
        className="quantity-input__button quantity-input__button--decrement"
        aria-label="Zmniejsz wartość"
      >
        −
      </button>

      <input
        type="text"
        value={value}
        onChange={handleInputChange}
        onBlur={handleBlur}
        className="quantity-input__field"
        aria-label="Ilość"
      />

      <button
        onClick={handleIncrement}
        disabled={value >= max}
        className="quantity-input__button quantity-input__button--increment"
        aria-label="Zwiększ wartość"
      >
        +
      </button>
    </div>
  );
};

export default QuantityInput;
