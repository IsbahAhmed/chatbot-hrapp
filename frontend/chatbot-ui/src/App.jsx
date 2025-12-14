
import './App.css'
import { useState, useEffect, useRef } from 'react'

function ErrorMessage({ content }) {
  this.content = content;
  this.role = "assistant";
  this.type = "error";
}

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMessage = { role: 'user', content: input };
    const filtered  = messages.filter(msg => !(msg.type && msg.type === "error"));
    const newMessages = [...filtered, userMessage];
    setMessages(newMessages);
    setInput('');
    try {
      const response = await fetch('http://localhost:8000/ask', {
        method: 'post',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: newMessages })
      });
      const data = await response.json();
      const botMessage = { role: 'assistant', content: data.reply };
      setMessages([...newMessages, botMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = new ErrorMessage({ content: 'Sorry, there was an error processing your message.' });
      setMessages([...newMessages, errorMessage]);
    }
  };

  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            {msg.content}
          </div>
        ))}
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
        {/* <button onClick={sendMessage}>Send</button> */}
      </div>
    </div>
  )
}

export default App
