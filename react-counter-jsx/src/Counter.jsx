import React, { useState } from "react";
import "./Counter.css";

/**
 * Counter component — demonstrates JSX and React useState hook.
 *
 * Features:
 *  - Increment / Decrement the count by 1
 *  - Increment / Decrement by a custom step value
 *  - Reset the counter back to zero
 */
function Counter() {
  const [count, setCount] = useState(0);
  const [step, setStep] = useState(1);

  const increment = () => setCount((prev) => prev + step);
  const decrement = () => setCount((prev) => prev - step);
  const reset = () => setCount(0);

  const handleStepChange = (e) => {
    const value = parseInt(e.target.value, 10);
    setStep(!isNaN(value) && value > 0 ? value : 1);
  };

  return (
    <div className="counter-container">
      <h1 className="counter-title">React JSX Counter</h1>

      <p className={`counter-display ${count > 0 ? "positive" : count < 0 ? "negative" : ""}`}>
        {count}
      </p>

      <div className="counter-controls">
        <button className="btn btn-decrement" onClick={decrement}>
          − {step}
        </button>
        <button className="btn btn-reset" onClick={reset}>
          Reset
        </button>
        <button className="btn btn-increment" onClick={increment}>
          + {step}
        </button>
      </div>

      <div className="step-control">
        <label htmlFor="step-input">Step size:</label>
        <input
          id="step-input"
          type="number"
          min="1"
          value={step}
          onChange={handleStepChange}
        />
      </div>
    </div>
  );
}

export default Counter;
