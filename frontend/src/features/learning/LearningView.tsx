import React, { useState } from 'react';

interface Adapter {
  id: string;
  name: string;
  type: string;
  domain: string;
  rank: number;
  status: 'active' | 'inactive';
}

export const LearningView: React.FC = () => {
  const [adapters, setAdapters] = useState<Adapter[]>([
    { id: '1', name: 'Urdu Skill Adapter', type: 'LoRA', domain: 'Urdu Language Translation', rank: 8, status: 'active' },
    { id: '2', name: 'Python Coding Skill Adapter', type: 'LoRA', domain: 'Coding & Logic Synthesis', rank: 16, status: 'active' },
    { id: '3', name: 'MMMU Vision Adapter', type: 'QLoRA', domain: 'Multimodal Spatial Reasoning', rank: 8, status: 'inactive' }
  ]);

  const [replayRatio, setReplayRatio] = useState(0.3);
  const [distillAlpha, setDistillAlpha] = useState(0.5);
  const [ewcWeight, setEwcWeight] = useState(1000.0);

  const [newAdapterName, setNewAdapterName] = useState('');
  const [newAdapterRank, setNewAdapterRank] = useState(8);

  const handleAddAdapter = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newAdapterName.trim()) return;

    const newAd: Adapter = {
      id: String(Date.now()),
      name: newAdapterName,
      type: 'LoRA',
      domain: 'Custom',
      rank: newAdapterRank,
      status: 'active'
    };

    setAdapters(prev => [...prev, newAd]);
    setNewAdapterName('');
  };

  const toggleAdapter = (id: string) => {
    setAdapters(prev => prev.map(a => a.id === id ? { ...a, status: a.status === 'active' ? 'inactive' : 'active' } : a));
  };

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6 text-slate-100 bg-slate-950 min-h-screen">
      <header className="border-b border-slate-800 pb-4">
        <h2 className="text-2xl font-black text-white">Three-Speed Learning Architecture</h2>
        <p className="text-xs text-slate-400 mt-1">Configure Fast (RAG/KG), Medium (Adapters), and Slow (EWC/Distillation) learning variables.</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* ADAPTERS */}
        <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-6">
          <h3 className="text-sm font-black text-slate-400 uppercase tracking-wider block">Specialized LoRA Adapters</h3>

          <form onSubmit={handleAddAdapter} className="bg-slate-950 p-4 rounded-xl border border-slate-850 flex items-end space-x-3">
            <div className="flex-grow space-y-1">
              <label className="text-[10px] text-slate-500 font-bold uppercase">New Adapter Name</label>
              <input
                type="text"
                value={newAdapterName}
                onChange={e => setNewAdapterName(e.target.value)}
                placeholder="e.g. Arabic Skill Adapter"
                className="w-full p-2.5 bg-slate-900 border border-slate-800 rounded text-xs text-white focus:outline-none"
              />
            </div>
            <div className="w-24 space-y-1">
              <label className="text-[10px] text-slate-500 font-bold uppercase">Rank (R)</label>
              <input
                type="number"
                value={newAdapterRank}
                onChange={e => setNewAdapterRank(Number(e.target.value))}
                className="w-full p-2.5 bg-slate-900 border border-slate-800 rounded text-xs text-white focus:outline-none font-bold text-center"
              />
            </div>
            <button type="submit" className="bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold px-4 py-2.5 rounded-lg transition-all">
              Initialize
            </button>
          </form>

          <div className="space-y-3">
            {adapters.map(ad => (
              <div key={ad.id} className="p-4 bg-slate-950 border border-slate-850 rounded-xl flex justify-between items-center">
                <div className="space-y-1 text-xs">
                  <p className="font-bold text-slate-200">{ad.name}</p>
                  <p className="text-[10px] text-slate-500 uppercase">Domain: {ad.domain} | Rank: {ad.rank}</p>
                </div>
                <button
                  onClick={() => toggleAdapter(ad.id)}
                  className={`px-3 py-1 rounded text-[10px] font-bold uppercase ${
                    ad.status === 'active' ? 'bg-indigo-950 text-indigo-400 border border-indigo-900' : 'bg-slate-800 text-slate-400'
                  }`}
                >
                  {ad.status === 'active' ? 'Deactivate' : 'Activate'}
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* CONTINUAL LEARNING CONTROLS */}
        <div className="lg:col-span-5 bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-6 text-xs">
          <h3 className="text-sm font-black text-slate-400 uppercase tracking-wider border-b border-slate-850 pb-2">Continual Trainer Settings (Slow Learning)</h3>

          <div className="space-y-5">
            <div className="space-y-1">
              <div className="flex justify-between font-bold">
                <span>Experience Replay Ratio</span>
                <span>{replayRatio}</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={replayRatio}
                onChange={e => setReplayRatio(Number(e.target.value))}
                className="w-full cursor-pointer h-1 bg-slate-800 rounded-lg appearance-none"
              />
            </div>

            <div className="space-y-1">
              <div className="flex justify-between font-bold">
                <span>Distillation Temperature (Alpha)</span>
                <span>{distillAlpha}</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                value={distillAlpha}
                onChange={e => setDistillAlpha(Number(e.target.value))}
                className="w-full cursor-pointer h-1 bg-slate-800 rounded-lg appearance-none"
              />
            </div>

            <div className="space-y-1">
              <div className="flex justify-between font-bold">
                <span>Elastic Weight Consolidation (EWC) Weight</span>
                <span>{ewcWeight}</span>
              </div>
              <input
                type="range"
                min="100.0"
                max="5000.0"
                step="100.0"
                value={ewcWeight}
                onChange={e => setEwcWeight(Number(e.target.value))}
                className="w-full cursor-pointer h-1 bg-slate-800 rounded-lg appearance-none"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
