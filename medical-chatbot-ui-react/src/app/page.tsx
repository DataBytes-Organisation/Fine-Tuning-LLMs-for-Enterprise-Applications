"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown"; // Import react-markdown
import remarkGfm from "remark-gfm"; // Import plugin for GitHub-flavored markdown
import { getChatResponse } from "./chatService"; // Import the chat service
import Image from 'next/image';

export default function Home() {
  const [messages, setMessages] = useState([
    { sender: "bot", text: "Hello! I'm your personalized medical assistant. How can I help you today?" },
  ]);
  const [input, setInput] = useState("");
  const chatContainerRef = useRef<HTMLDivElement>(null);

  const handleSend = async () => {
    if (input.trim() === "") return;

    const userMessage = input;
    setMessages((prev) => [...prev, { sender: "user", text: userMessage }]);
    setInput("");

    // Get the bot's response using the chat service
    const botResponse = await getChatResponse(userMessage);
    setMessages((prev) => [...prev, { sender: "bot", text: botResponse }]);
  };

  const handleClearChat = () => {
    // Reset the chat to the initial state
    setMessages([{ sender: "bot", text: "Hello! I'm your personalized medical assistant. How can I help you today?" }]);
  };

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header Banner */}
      <header className="bg-gray-800 text-white py-4 px-8">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center gap-4">
            <Image src="/globe.svg" alt="Medical Assistant Icon" width={32} height={32} className="h-8 w-8" />
            <h1 className="text-2xl font-bold">DataBytes Medical Chatbot</h1>
          </div>
          <nav className="flex gap-4">
            <a href="#features" className="hover:underline">
              Features
            </a>
            <a href="#footer" className="hover:underline">
              Contact
            </a>
          </nav>
        </div>
      </header>

      {/* Hero Section */}
      <section className="bg-gray-100 py-10 px-8">
        <div className="max-w-7xl mx-auto text-center">
          <h2 className="text-4xl font-bold mb-4 text-gray-800">Welcome to Your Medical Assistant</h2>
          <p className="text-lg text-gray-700 mb-8">
            Get personalized medical advice and support from our intelligent chatbot. Designed to assist you with care and precision.
          </p>
        </div>
      </section>

      {/* Chatbox Section */}
      <section id="chat" className="py-16 px-8 bg-white">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col gap-4 w-full border border-gray-300 rounded-lg p-4 bg-gray-50">
            <div
              ref={chatContainerRef}
              className="flex flex-col gap-2 overflow-y-auto max-h-96"
            >
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`p-2 rounded-lg ${
                    message.sender === "bot"
                      ? "bg-gray-200 text-left"
                      : "bg-blue-100 text-right"
                  }`}
                >
                  {message.sender === "bot" ? (
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={{
                        h1: ({ ...props }) => (
                          <h1 className="text-2xl font-bold" {...props} />
                        ),
                        h2: ({ ...props }) => (
                          <h2 className="text-xl font-bold" {...props} />
                        ),
                        ul: ({ ...props }) => (
                          <ul className="list-disc ml-5" {...props} />
                        ),
                        ol: ({ ...props }) => (
                          <ol className="list-decimal ml-5" {...props} />
                        ),
                        p: ({ ...props }) => (
                          <p className="text-gray-800" {...props} />
                        ),
                      }}
                    >
                      {message.text}
                    </ReactMarkdown>
                  ) : (
                    message.text
                  )}
                </div>
              ))}
            </div>
            <div className="flex gap-2 mt-4">
              <input
                type="text"
                className="flex-1 border border-gray-300 rounded-lg p-2 bg-gray-50 text-gray-800"
                placeholder="Type your message..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSend()}
              />
              <button
                className="bg-gray-800 text-white px-4 py-2 rounded-lg hover:bg-gray-900"
                onClick={handleSend}
              >
                Send
              </button>
              <button
                className="bg-red-500 text-white px-4 py-2 rounded-lg hover:bg-red-600"
                onClick={handleClearChat}
              >
                Clear Chat
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="bg-gray-100 py-16 px-8">
        <div className="max-w-7xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-8 text-gray-800">Features</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
            <div className="p-6 bg-white rounded-lg shadow-md border border-gray-300">
              <h3 className="text-xl font-bold mb-2 text-gray-800">Personalised Medical Advice</h3>
              <p className="text-gray-700">
                Receive tailored responses to your medical queries based on your input.
              </p>
            </div>
            <div className="p-6 bg-white rounded-lg shadow-md border border-gray-300">
              <h3 className="text-xl font-bold mb-2 text-gray-800">User-Friendly Interface</h3>
              <p className="text-gray-700">
                Navigate effortlessly with our clean and intuitive design.
              </p>
            </div>
            <div className="p-6 bg-white rounded-lg shadow-md border border-gray-300">
              <h3 className="text-xl font-bold mb-2 text-gray-800">Secure and Confidential</h3>
              <p className="text-gray-700">
                Your conversations are private and secure, ensuring peace of mind.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer id="footer" className="bg-gray-800 text-white py-8 px-8 mt-auto">
        <div className="max-w-7xl mx-auto text-center">
          <p className="text-lg font-bold">Medical Chatbot</p>
          <p className="text-sm mt-2">
            © {new Date().getFullYear()} DataBytes. All rights reserved.
          </p>
          <p className="mt-4">
            Built with ❤️ by{" "}
            <a
              href="https://github.com/digitalblue"
              target="_blank"
              rel="noopener noreferrer"
              className="underline hover:text-gray-300"
            >
              Ed Ras
            </a>
          </p>
        </div>
      </footer>
    </div>
  );
}
