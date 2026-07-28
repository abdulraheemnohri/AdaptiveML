import React, { useState } from 'react';

export const SettingsPage: React.FC = () => {
  const [temperature, setTemperature] = useState(0.7);
  const [precision, setPrecision] = useState('bf16');

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 bg-slate-900 border border-slate-800 rounded-2xl p-6 text-slate-100">
      <div className="space-y-4">
        <h2 className="text-sm font-black text-slate-400 uppercase tracking-wider border-b border-slate-800 pb-3">Inference Parameters</h2>
        <div className="space-y-2 text-xs">
          <div className="flex justify-between font-bold">
            <span>Temperature</span>
            <span>{temperature}</span>
          </div>
          <input type="range" min="0.1" max="1.5" step="0.05" value={temperature} onChange={e => setTemperature(Number(e.target.value))} className="w-full cursor-pointer" />
        </div>
      </div>

      <div className="space-y-4">
        <h3 className="text-sm font-black text-slate-400 uppercase tracking-wider border-b border-slate-800 pb-3">Compute & Precision</h3>
        <div className="grid grid-cols-3 gap-2">
          {['bf16', 'fp16', 'fp32'].map(p => (
            <button key={p} onClick={() => setPrecision(p)} className={`p-2.5 rounded text-xs font-bold uppercase transition ${precision === p ? 'bg-indigo-600 text-white' : 'bg-slate-800 text-slate-400'}`}>{p}</button>
          ))}
        </div>
      </div>
    </div>
  );
};
