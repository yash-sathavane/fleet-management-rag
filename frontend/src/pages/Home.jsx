import Navbar from "../components/Navbar";
import Sidebar from "../components/Sidebar";
import ChatWindow from "../components/ChatWindow";

function Home() {

    return (

        <div className="min-h-screen bg-slate-900">

            <Navbar />

            <div className="flex h-[calc(100vh-64px)]">

                {/* <Sidebar /> */}

                <ChatWindow />

            </div>

        </div>

    );

}

export default Home;