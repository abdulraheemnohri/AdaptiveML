import React, { useState } from 'react';

interface DataSource {
  id: string;
  name: string;
  type: string;
}

export const DatasetManagerPage: React.FC = () => {
  const [sources, setSources] = useState<DataSource[]>([
    { id: '1', name: 'Urdu Lexical sitemap', type: 'Sitemap' },
    { id: '2', name: 'GitHub MMMU vision repo', type: 'Git Repository' }
  ]);
  const [newSourceName, setNewSourceName] = useState("");

  const handleAddSource = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newSourceName.trim()) return;

    setSources(prev => [...prev, { id: String(Date.now()), name: newSourceName, type: 'Website' }]);
    setNewSourceName("");
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 text-slate-100 space-y-6">
      <h2 className="text-sm font-black text-slate-400 uppercase tracking-wider border-b border-slate-800 pb-3">Dataset & Ingestion Manager</h2>
      <form onSubmit={handleAddSource} className="flex space-x-2">
        <input type="text" value={newSourceName} onChange={e => setNewSourceName(e.target.value)} placeholder="Dataset Source URL..." className="flex-grow p-2.5 bg-slate-950 border border-slate-800 rounded text-xs focus:outline-none" />
        <button type="submit" className="bg-indigo-600 px-4 rounded text-xs font-bold text-white">Register Source</button>
      </form>
      <div className="space-y-2">
        {sources.map(src => (
          <div key={src.id} className="p-3 bg-slate-950 border border-slate-850 rounded-xl flex justify-between items-center text-xs">
            <span className="font-bold text-slate-200">{src.name}</span>
            <span className="text-[10px] bg-slate-800 text-slate-400 px-2 py-0.5 rounded font-black">{src.type}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
