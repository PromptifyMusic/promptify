import React, { useState, useEffect } from 'react';

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
    <div className={`flex items-center gap-0 ${className}`}>
      <button
        onClick={handleDecrement}
        disabled={value <= min}
        className="
          w-10 h-10 
          bg-gray-700 hover:bg-gray-600 
          disabled:bg-gray-800 disabled:text-gray-600 disabled:cursor-not-allowed
          text-white font-bold text-xl
          rounded-l-lg
          transition-colors duration-200
          flex items-center justify-center
          focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900
        "
        aria-label="Zmniejsz wartość"
      >
        −
      </button>

      <input
        type="text"
        value={value}
        onChange={handleInputChange}
        onBlur={handleBlur}
        className="
          w-16 h-10
          bg-gray-800 
          text-white text-center font-semibold
          border-y-2 border-gray-700
          focus:outline-none focus:ring-2 focus:ring-blue-500 focus:z-10
          transition-all duration-200
        "
        aria-label="Ilość"
      />

      <button
        onClick={handleIncrement}
        disabled={value >= max}
        className="
          w-10 h-10
          bg-gray-700 hover:bg-gray-600
          disabled:bg-gray-800 disabled:text-gray-600 disabled:cursor-not-allowed
          text-white font-bold text-xl
          rounded-r-lg
          transition-colors duration-200
          flex items-center justify-center
          focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900
        "
        aria-label="Zwiększ wartość"
      >
        +
      </button>
    </div>
  );
};

export default QuantityInput;
