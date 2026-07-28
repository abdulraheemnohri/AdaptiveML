import React, { useState } from 'react';

interface KnowledgeGap {
  id: string;
  topic: string;
  importance: string;
  status: string;
}

export const GapResearchPage: React.FC = () => {
  const [gaps, setGaps] = useState<KnowledgeGap[]>([
    { id: '1', topic: 'Urdu colloquial dialect translations', importance: 'high', status: 'researching' },
    { id: '2', topic: 'MMMU Video Frame Spatial Temporal Reasoning', importance: 'critical', status: 'queued' }
  ]);
  const [newGap, setNewGap] = useState("");

  const handleAddGap = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newGap.trim()) return;

    setGaps(prev => [...prev, { id: String(Date.now()), topic: newGap, importance: 'high', status: 'queued' }]);
    setNewGap("");
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 text-slate-100 space-y-6 text-xs">
      <h2 className="text-sm font-black text-slate-400 uppercase tracking-wider border-b border-slate-800 pb-3">Knowledge Gap Discovery & Autonomous Agents</h2>

      <form onSubmit={handleAddGap} className="flex space-x-2">
        <input type="text" value={newGap} onChange={e => setNewGap(e.target.value)} placeholder="Register new knowledge gap..." className="flex-grow p-2.5 bg-slate-950 border border-slate-800 rounded text-xs focus:outline-none text-white" />
        <button type="submit" className="bg-indigo-600 px-4 rounded text-xs font-bold text-white">Register</button>
      </form>

      <div className="space-y-2">
        {gaps.map(gap => (
          <div key={gap.id} className="p-3 bg-slate-950 border border-slate-850 rounded-xl flex justify-between items-center text-xs">
            <div>
              <p className="font-bold text-slate-200">{gap.topic}</p>
              <p className="text-[10px] text-red-400 uppercase font-black">Importance: {gap.importance}</p>
            </div>
            <span className="text-cyan-400 uppercase font-black text-[10px]">{gap.status}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
