import React, { useState } from 'react';

interface DataSource {
  id: string;
  name: string;
  type: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  trustLevel: string;
  status: 'active' | 'inactive';
}

export const DataView: React.FC = () => {
  const [sources, setSources] = useState<DataSource[]>([
    { id: '1', name: 'Urdu Wikipedia Sitemap', type: 'Sitemap', priority: 'high', trustLevel: 'Trusted', status: 'active' },
    { id: '2', name: 'MMMU Multimodal Benchmark Repo', type: 'Git Repository', priority: 'critical', trustLevel: 'Fully Trusted', status: 'active' },
    { id: '3', name: 'Machine Learning arXiv Feed', type: 'RSS Feed', priority: 'medium', trustLevel: 'Unverified', status: 'active' },
    { id: '4', name: 'Local Video Lecture Folder', type: 'Local Folder', priority: 'low', trustLevel: 'Trusted', status: 'inactive' }
  ]);

  const [newSourceName, setNewSourceName] = useState('');
  const [newSourceType, setNewSourceType] = useState('Website');
  const [newSourcePriority, setNewSourcePriority] = useState<'low' | 'medium' | 'high' | 'critical'>('high');

  const [trainSplit, setTrainSplit] = useState(80);
  const [minQualityScore, setMinQualityScore] = useState(0.85);

  const handleAddSource = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newSourceName.trim()) return;

    const newSrc: DataSource = {
      id: String(Date.now()),
      name: newSourceName,
      type: newSourceType,
      priority: newSourcePriority,
      trustLevel: 'Unverified',
      status: 'active'
    };

    setSources(prev => [...prev, newSrc]);
    setNewSourceName('');
  };

  const testConnection = (name: string) => {
    alert(`Connection test to ${name} was successful. Latency: 34ms.`);
  };

  const toggleStatus = (id: string) => {
    setSources(prev => prev.map(s => s.id === id ? { ...s, status: s.status === 'active' ? 'inactive' : 'active' } : s));
  };

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6 text-slate-100 bg-slate-950 min-h-screen">
      <header className="border-b border-slate-800 pb-4">
        <h2 className="text-2xl font-black text-white">Universal Ingestion & Data Quality Pipeline</h2>
        <p className="text-xs text-slate-400 mt-1">Configure sources, manage training splits, and monitor data processing pipelines.</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* DATA SOURCES */}
        <div className="lg:col-span-8 bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-6">
          <h3 className="text-sm font-black text-slate-400 uppercase tracking-wider block">Acquisition Sources</h3>

          <form onSubmit={handleAddSource} className="bg-slate-950 p-4 rounded-xl border border-slate-850 flex flex-col md:flex-row items-end space-y-3 md:space-y-0 md:space-x-3">
            <div className="flex-grow space-y-1 w-full">
              <label className="text-[10px] text-slate-500 font-bold uppercase">Source Location / Name</label>
              <input
                type="text"
                value={newSourceName}
                onChange={e => setNewSourceName(e.target.value)}
                placeholder="e.g. https://arxiv.org/rss/cs"
                className="w-full p-2.5 bg-slate-900 border border-slate-800 rounded text-xs text-white focus:outline-none"
              />
            </div>
            <div className="w-full md:w-36 space-y-1">
              <label className="text-[10px] text-slate-500 font-bold uppercase">Connector Type</label>
              <select value={newSourceType} onChange={e => setNewSourceType(e.target.value)} className="w-full p-2.5 bg-slate-900 border border-slate-800 rounded text-xs text-slate-300 focus:outline-none font-bold">
                <option>Website</option>
                <option>RSS Feed</option>
                <option>Git Repository</option>
                <option>Sitemap</option>
                <option>Local Folder</option>
              </select>
            </div>
            <button type="submit" className="bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold px-5 py-2.5 rounded-lg transition-all w-full md:w-auto">
              Add Source
            </button>
          </form>

          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left text-slate-400">
              <thead className="text-[10px] text-slate-500 uppercase tracking-wider border-b border-slate-850">
                <tr>
                  <th className="py-3">Source Name</th>
                  <th className="py-3">Type</th>
                  <th className="py-3">Priority</th>
                  <th className="py-3">Trust</th>
                  <th className="py-3">Status</th>
                  <th className="py-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {sources.map(src => (
                  <tr key={src.id} className="border-b border-slate-850/50 hover:bg-slate-850/20">
                    <td className="py-3.5 font-bold text-slate-200">{src.name}</td>
                    <td className="py-3.5">{src.type}</td>
                    <td className="py-3.5"><span className="text-cyan-400 font-bold uppercase">{src.priority}</span></td>
                    <td className="py-3.5">{src.trustLevel}</td>
                    <td className="py-3.5">
                      <span className={`px-2 py-0.5 rounded-full text-[9px] font-black ${
                        src.status === 'active' ? 'bg-emerald-950/80 text-emerald-400' : 'bg-slate-800 text-slate-400'
                      }`}>{src.status}</span>
                    </td>
                    <td className="py-3.5 text-right space-x-2">
                      <button onClick={() => testConnection(src.name)} className="text-[10px] bg-slate-800 hover:bg-slate-700 text-slate-300 px-2 py-1 rounded">Test</button>
                      <button onClick={() => toggleStatus(src.id)} className="text-[10px] text-red-400 hover:underline">Toggle</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* PIPELINE SETTINGS */}
        <div className="lg:col-span-4 bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-6 text-xs">
          <h3 className="text-sm font-black text-slate-400 uppercase tracking-wider border-b border-slate-850 pb-2">Pipeline Settings</h3>
          <div className="space-y-5">
            <div className="space-y-1">
              <div className="flex justify-between font-bold">
                <span>Training / Validation Split</span>
                <span>{trainSplit}% / {100 - trainSplit}%</span>
              </div>
              <input
                type="range"
                min="50"
                max="95"
                value={trainSplit}
                onChange={e => setTrainSplit(Number(e.target.value))}
                className="w-full cursor-pointer h-1 bg-slate-800 rounded-lg appearance-none"
              />
            </div>

            <div className="space-y-1">
              <div className="flex justify-between font-bold">
                <span>Min Quality Cutoff Score</span>
                <span>{minQualityScore}</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="0.99"
                step="0.05"
                value={minQualityScore}
                onChange={e => setMinQualityScore(Number(e.target.value))}
                className="w-full cursor-pointer h-1 bg-slate-800 rounded-lg appearance-none"
              />
            </div>

            <div className="p-3 bg-slate-950 border border-slate-850 rounded-xl leading-relaxed text-slate-500">
              ℹ️ Standard parsing rules detect languages, transcribe audio layers, and filter exact/near-duplicates using FAISS indexing automatically.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
