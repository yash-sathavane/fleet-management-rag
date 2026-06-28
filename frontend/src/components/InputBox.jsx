function InputBox({ question, setQuestion, onSend }) {

    return (

        <div className="flex gap-3 p-4 border-t border-slate-700">

            <input

                className="flex-1 bg-slate-800 rounded-lg p-3 text-white outline-none"

                placeholder="Ask something..."

                value={question}

                onChange={(e) => setQuestion(e.target.value)}

            />

            <button

                onClick={onSend}

                className="bg-cyan-500 px-6 rounded-lg hover:bg-cyan-600"

            >

                Send

            </button>

        </div>

    );

}

export default InputBox;