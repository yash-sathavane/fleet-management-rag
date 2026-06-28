function Message({ text, sender }) {

    const isUser = sender === "user";

    return (

        <div
            className={`flex ${
                isUser ? "justify-end" : "justify-start"
            } mb-4`}
        >

            <div
                className={`max-w-[70%] rounded-xl px-4 py-3 ${
                    isUser
                        ? "bg-cyan-500 text-white"
                        : "bg-slate-700 text-white"
                }`}
            >

                {text}

            </div>

        </div>

    );

}

export default Message;