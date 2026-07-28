import React, { useState } from 'react';

interface MemoryEntry {
  id: string;
  type: string;
  content: string;
}

export const MemoryPage: React.FC = () => {
  const [memories, setMemories] = useState<MemoryEntry[]>([
    { id: '1', type: 'user', content: 'Prefers concise documentation with detailed python snippets.' },
    { id: '2', type: 'task', content: 'Continual learning checkpoint sync triggers.' }
  ]);
  const [newMem, setNewMem] = useState("");
  const [chunkSize, setChunkSize] = useState(512);

  const handleAddMemory = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newMem.trim()) return;

    setMemories(prev => [...prev, { id: String(Date.now()), type: 'user', content: newMem }]);
    setNewMem("");
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 text-slate-100 bg-slate-900 border border-slate-800 rounded-2xl p-6">
      <div className="space-y-6">
        <h2 className="text-sm font-black text-slate-400 uppercase tracking-wider border-b border-slate-800 pb-3">Long-Term Memory Console</h2>
        <form onSubmit={handleAddMemory} className="flex space-x-2">
          <input type="text" value={newMem} onChange={e => setNewMem(e.target.value)} placeholder="Add memory..." className="flex-grow p-2.5 bg-slate-950 border border-slate-800 rounded text-xs focus:outline-none" />
          <button type="submit" className="bg-indigo-600 px-4 rounded text-xs font-bold text-white">Add</button>
        </form>
        <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
          {memories.map(m => (
            <div key={m.id} className="p-3 bg-slate-950 border border-slate-850 rounded-xl flex justify-between items-center text-xs">
              <span className="text-slate-300">{m.content}</span>
              <button onClick={() => setMemories(prev => prev.filter(x => x.id !== m.id))} className="text-red-400 hover:underline">Forget</button>
            </div>
          ))}
        </div>
      </div>

      <div className="space-y-4">
        <h3 className="text-sm font-black text-slate-400 uppercase tracking-wider border-b border-slate-800 pb-3">RAG Similarity Config</h3>
        <div className="space-y-2 text-xs">
          <div className="flex justify-between font-bold">
            <span>RAG Chunk Size</span>
            <span>{chunkSize} tokens</span>
          </div>
          <input type="range" min="128" max="1024" step="128" value={chunkSize} onChange={e => setChunkSize(Number(e.target.value))} className="w-full cursor-pointer" />
        </div>
      </div>
    </div>
  );
};
