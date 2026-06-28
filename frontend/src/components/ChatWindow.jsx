import { useState } from "react";
import Message from "./Message";
import InputBox from "./InputBox";
import api from "../services/api";

function ChatWindow() {

    const [messages, setMessages] = useState([
        {
            sender: "ai",
            text: "Hello 👋 I am your Fleet AI Assistant."
        }
    ]);

    const [question, setQuestion] = useState("");

    async function sendMessage() {

    if (question.trim() === "") return;

    const userMessage = {
        sender: "user",
        text: question
    };

    setMessages((prev) => [...prev, userMessage]);

    const currentQuestion = question;

    setQuestion("");

    try {

        const response = await api.post("/ask", {
            question: currentQuestion
        });

        const aiMessage = {
            sender: "ai",
            text: response.data.answer
        };

        setMessages((prev) => [...prev, aiMessage]);

    }

    catch (error) {

        console.error(error);

        setMessages((prev) => [
            ...prev,
            {
                sender: "ai",
                text: "Unable to connect to server."
            }
        ]);

    }

}

    return (

        <div className="flex flex-col flex-1 bg-slate-950" >

            <div className="flex-1 p-6 overflow-y-auto">

                <h2 className="text-2xl font-bold text-cyan-400 mb-6">

                    AI Assistant

                </h2>

                {
                    messages.map((message, index) => (

                        <Message

                            key={index}

                            sender={message.sender}

                            text={message.text}

                        />

                    ))
                }

            </div>

         <div className="p-4 border-t border-slate-800 mt-2" >     
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