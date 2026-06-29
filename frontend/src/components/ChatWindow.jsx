
import { useState } from "react";
import Message from "./Message";
import InputBox from "./InputBox";
import api from "../services/api";
import { useTheme } from "../context/ThemeContext";

function ChatWindow() {
  const { darkMode } = useTheme();

  const [messages, setMessages] = useState([
    {
      sender: "ai",
      text: "Hello 👋 I am your Fleet AI Assistant.",
    },
  ]);

  const [question, setQuestion] = useState("");

  async function sendMessage() {
    if (question.trim() === "") return;

    const userMessage = {
      sender: "user",
      text: question,
    };

    setMessages((prev) => [...prev, userMessage]);

    const currentQuestion = question;
    setQuestion("");

    try {
      const response = await api.post("/ask", {
        question: currentQuestion,
      });

      const aiMessage = {
        sender: "ai",
        text: response.data.answer,
      };

      setMessages((prev) => [...prev, aiMessage]);
    } catch (error) {
      console.error(error);

      setMessages((prev) => [
        ...prev,
        {
          sender: "ai",
          text: "Unable to connect to server.",
        },
      ]);
    }
  }

  return (
    <div
      className={`flex flex-col flex-1 transition-all duration-300 ${
        darkMode ? "bg-slate-950" : "bg-gray-100"
      }`}
    >
      <div className="flex-1 p-6 overflow-y-auto">
        <h2
          className={`text-2xl font-bold mb-6 ${
            darkMode ? "text-cyan-400" : "text-gray-900"
          }`}
        >
          AI Assistant
        </h2>

        {messages.map((message, index) => (
          <Message
            key={index}
            sender={message.sender}
            text={message.text}
          />
        ))}
      </div>

      <div
        className={`p-4 border-t transition-all duration-300 ${
          darkMode
            ? "border-slate-700 bg-slate-900"
            : "border-gray-300 bg-white"
        }`}
      >
        <InputBox
          question={question}
          setQuestion={setQuestion}
          onSend={sendMessage}
        />
      </div>
    </div>
  );
}

export default ChatWindow;