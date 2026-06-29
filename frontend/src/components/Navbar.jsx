

import { useTheme } from "../context/ThemeContext";

function Navbar() {
  const { darkMode, setDarkMode } = useTheme();

  return (
    <nav
      className={`h-16 flex justify-between items-center px-6 shadow-md transition-all duration-300
      ${
        darkMode
          ? "bg-slate-900 border-b border-slate-700"
          : "bg-white border-b border-gray-300"
      }`}
    >
      <h1
        className={`text-2xl font-bold ${
          darkMode ? "text-cyan-400" : "text-gray-900"
        }`}
      >
        🚚 Fleet Management AI
      </h1>

      <button
        onClick={() => setDarkMode(!darkMode)}
        className={`px-4 py-2 rounded-lg font-medium transition
        ${
          darkMode
            ? "bg-yellow-400 text-black hover:bg-yellow-300"
            : "bg-slate-800 text-white hover:bg-slate-700"
        }`}
      >
        {darkMode ? "☀ Light" : "🌙 Dark"}
      </button>
    </nav>
  );
}

export default Navbar;