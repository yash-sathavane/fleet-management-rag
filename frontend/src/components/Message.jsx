

import { useTheme } from "../context/ThemeContext";

function Message({ text, sender }) {
  const { darkMode } = useTheme();

  const isUser = sender === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div
        className={`max-w-[70%] rounded-xl px-4 py-3 shadow-sm
        ${
          isUser
            ? "bg-cyan-600 text-white"
            : darkMode
            ? "bg-slate-700 text-white"
            : "bg-white text-gray-900 border border-gray-300"
        }`}
      >
        {text}
      </div>
    </div>
  );
}

export default Message;