# ⚛️ React JSX Counter

A simple, interactive **counter application** built with **React** and **JSX** that demonstrates core React concepts including functional components, the `useState` hook, and event handling.

---

## 📸 Features

| Feature | Description |
|---------|-------------|
| **Increment** | Increase the counter by the configured step |
| **Decrement** | Decrease the counter by the configured step |
| **Reset** | Reset the counter back to zero |
| **Custom Step** | Change the step size to increment/decrement by any positive integer |
| **Color Feedback** | Counter turns green for positive values, red for negative values |

---

## 🗂️ Project Structure

```
react-counter-jsx/
├── public/
│   └── index.html          # HTML template
├── src/
│   ├── index.jsx            # React entry point — mounts the app
│   ├── App.jsx              # Root component
│   ├── Counter.jsx          # Counter component (logic + JSX)
│   └── Counter.css          # Counter styles
├── .gitignore
├── package.json
├── webpack.config.js
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) v16 or higher
- npm (bundled with Node.js)

### Installation

```bash
# Navigate into the project directory
cd react-counter-jsx

# Install dependencies
npm install
```

### Run the Development Server

```bash
npm start
```

The app will open automatically at **http://localhost:3000**.

### Production Build

```bash
npm run build
```

The optimised output is placed in the `dist/` folder.

---

## 🧑‍💻 Key Concepts Demonstrated

### Functional Component with JSX

```jsx
import React, { useState } from "react";

function Counter() {
  const [count, setCount] = useState(0);

  return (
    <div>
      <p>{count}</p>
      <button onClick={() => setCount(count + 1)}>+</button>
    </div>
  );
}
```

### `useState` Hook

The `useState` hook manages two pieces of state:
- `count` — the current counter value (initial value `0`)
- `step`  — the increment/decrement step size (initial value `1`)

### Event Handling

Each button calls a handler that uses the functional form of `setCount` to safely derive the next state from the previous state:

```jsx
const increment = () => setCount((prev) => prev + step);
const decrement = () => setCount((prev) => prev - step);
const reset     = () => setCount(0);
```

---

## 🛠️ Technologies

- **React 18** — UI library
- **JSX** — JavaScript XML syntax
- **Webpack 5** — module bundler
- **Babel** — JSX & ES2015+ transpiler
- **CSS** — component styling

---

## 📄 License

MIT
