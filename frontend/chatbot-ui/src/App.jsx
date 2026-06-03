
import './App.css'
import { useState, useEffect, useRef } from 'react'
import { ShimmerText } from "react-shimmer-effects";

function ErrorMessage({ content }) {
  this.content = content;
  this.role = "assistant";
  this.type = "error";
}

function getSessionId() {
  const key = 'hr_chat_session_id';
  let id = localStorage.getItem(key);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(key, id);
  }
  return id;
}

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);
  const [loading, setLoading] = useState(false);
  const sessionIdRef = useRef(getSessionId());

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim()) return;
    setLoading(true);
    const text = input.trim();
    const userMessage = { role: 'user', content: text };
    const filtered  = messages.filter(msg => !(msg.type && msg.type === "error"));
    const newMessages = [...filtered, userMessage];
    setMessages(newMessages);
    setInput('');
    try {
      const response = await fetch('http://localhost:8000/ask', {
        method: 'post',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          session_id: sessionIdRef.current,
        })
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Request failed');
      }
      const botMessage = { role: 'assistant', content: data.reply };
      setMessages([...newMessages, botMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = new ErrorMessage({ content: 'Sorry, there was an error processing your message.' });
      setMessages([...newMessages, errorMessage]);
    }
    finally {
      setLoading(false)
    }
  };

  const startNewChat = async () => {
    const oldSessionId = sessionIdRef.current;
    try {
      await fetch('http://localhost:8000/session/reset', {
        method: 'post',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: oldSessionId }),
      });
    } catch (e) {
      console.warn('Could not reset server session', e);
    }
    localStorage.removeItem('hr_chat_session_id');
    sessionIdRef.current = getSessionId();
    setMessages([]);
    setInput('');
  };

  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            {msg.content}
          </div>
        ))}
        {
          loading && (
            <div className="loader">
              <ShimmerText line={3} gap={10} />
            </div>
          )
        }
        <div ref={messagesEndRef} />
      </div>
      <div className="input-area">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Type your message..."
        />
        <button type="button" onClick={startNewChat} title="Start new conversation">
          New chat
        </button>
      </div>
    </div>
  )
}

export default App
