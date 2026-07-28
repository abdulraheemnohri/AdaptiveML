import React, { useState } from 'react';

interface ChatMessage {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  modality: string;
  timestamp: string;
  explanation?: string;
  rating?: number;
}

export const AIWorkspacePage: React.FC = () => {
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      id: "msg-1",
      sender: "assistant",
      text: "Greetings! This is Qwen2.5-Omni-3B AI Workspace. Upload images, audio, video, or documents and ask me anything.",
      modality: "text",
      timestamp: new Date().toLocaleTimeString()
    }
  ]);
  const [inputText, setInputText] = useState("");
  const [activeModality, setActiveModality] = useState("text");
  const [selectedModel, setSelectedModel] = useState("Qwen/Qwen2.5-Omni-3B");
  const [showExplanationId, setShowExplanationId] = useState<string | null>(null);

  const handleSendMessage = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim()) return;

    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      sender: 'user',
      text: inputText,
      modality: activeModality,
      timestamp: new Date().toLocaleTimeString()
    };
    setChatMessages(prev => [...prev, userMsg]);
    setInputText("");

    setTimeout(() => {
      const botMsg: ChatMessage = {
        id: `bot-${Date.now()}`,
        sender: 'assistant',
        text: `Completed inference using ${selectedModel} model. Context: '${userMsg.text}' processed successfully in ${activeModality} mode.`,
        modality: activeModality,
        timestamp: new Date().toLocaleTimeString(),
        explanation: "Inference routed through core multimodal blocks. Memory regularisation was applied to retain Dialect and Code capabilities."
      };
      setChatMessages(prev => [...prev, botMsg]);
    }, 600);
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 flex flex-col h-[650px] text-slate-100">
      <div className="flex justify-between items-center border-b border-slate-800 pb-4 mb-4">
        <h2 className="text-sm font-black text-slate-400 uppercase tracking-wider flex items-center">
          💬 Multimodal Workspace Page
        </h2>
        <div className="flex space-x-1">
          {['text', 'image', 'audio', 'video', 'speech'].map(m => (
            <button key={m} onClick={() => setActiveModality(m)} className={`px-3 py-1 rounded text-[10px] font-bold uppercase transition ${activeModality === m ? 'bg-indigo-600 text-white' : 'bg-slate-800 text-slate-400 hover:bg-slate-700'}`}>{m}</button>
          ))}
        </div>
      </div>

      <div className="flex-grow overflow-y-auto space-y-4 mb-4 pr-1">
        {chatMessages.map(msg => (
          <div key={msg.id} className={`flex flex-col ${msg.sender === 'user' ? 'items-end' : 'items-start'}`}>
            <div className={`p-4 rounded-2xl max-w-xl text-xs leading-relaxed ${msg.sender === 'user' ? 'bg-indigo-600 text-white rounded-tr-none' : 'bg-slate-950 border border-slate-850 text-slate-300 rounded-tl-none'}`}>
              <div className="whitespace-pre-wrap">{msg.text}</div>
            </div>
            {msg.sender === 'assistant' && (
              <div className="flex space-x-3 text-[10px] mt-1.5 font-bold text-slate-400">
                {msg.explanation && <button onClick={() => setShowExplanationId(showExplanationId === msg.id ? null : msg.id)} className="text-indigo-400 hover:underline">💡 Explain</button>}
                <button onClick={() => alert("Marked correct")} className="hover:text-emerald-400">👍 Correct</button>
              </div>
            )}
            {showExplanationId === msg.id && msg.explanation && (
              <div className="mt-2 p-3 bg-slate-950 border border-indigo-950 text-indigo-300 rounded-xl text-[10px] max-w-xl">{msg.explanation}</div>
            )}
          </div>
        ))}
      </div>

      <form onSubmit={handleSendMessage} className="flex space-x-2">
        <input type="text" value={inputText} onChange={e => setInputText(e.target.value)} placeholder="Type a message..." className="flex-grow p-3 bg-slate-950 border border-slate-800 rounded-xl text-xs text-white focus:outline-none" />
        <button type="submit" className="bg-indigo-600 hover:bg-indigo-700 text-white px-5 rounded-xl text-xs font-bold">Send</button>
      </form>
    </div>
  );
};
