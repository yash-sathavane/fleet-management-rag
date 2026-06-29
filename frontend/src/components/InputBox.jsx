

import { useTheme } from "../context/ThemeContext";

function InputBox({ question, setQuestion, onSend }) {
  const { darkMode } = useTheme();

  return (
    <div className="flex gap-3">
      <input
        type="text"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && onSend()}
        placeholder="Ask something..."
        className={`flex-1 rounded-lg p-3 border outline-none transition
        ${
          darkMode
            ? "bg-slate-800 border-slate-700 text-white placeholder-gray-400"
            : "bg-white border-gray-300 text-gray-900 placeholder-gray-500"
        }`}
      />

      <button
        onClick={onSend}
        className="bg-cyan-600 text-white px-6 rounded-lg hover:bg-cyan-700 transition"
      >
        Send
      </button>
    </div>
  );
}

export default InputBox;